"""Chat router: question -> structured Query -> safe SQL -> answer "blocks" the UI can draw.

Flow (no model is needed for any number you see):
  1. ``chat_parser.parse`` turns the question into a ``Query`` (intent + filters) with fixed rules.
  2. This module runs the Query with validated, parameterised SQL from ``analytics`` / ``database``.
  3. For "why / reasons / summarise" questions an optional ``summarizer`` (an LLM, added later) may only *word*
     the retrieved evidence. Without one, or if it fails, a rule-based summary built from the same numbers is shown.
The router never turns user text into SQL and never reads the ``extra`` columns, so email/phone-like data cannot
reach an answer or a prompt. Every answer says how the question was understood, so a wrong reading is visible.

``route`` returns {"blocks": [...], "mode": None | "add_review"}. Block types:
text, breakdown, chart, table, reviews, evidence, badge, route, actions, sentiment.
"""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable

import config
import database
from services import analytics, chat_parser, product_change
from services import insights
from services import topics as topics_service
from services.chat_parser import DEFAULT_LIMIT, Query

# summarizer(facts, examples) -> text, or None/raise when unavailable
Summarizer = Callable[[dict[str, Any], list[dict[str, Any]]], "str | None"]

HELP_TEXT = (
    "I answer questions about the loaded feedback, and I only use what is in the data. Examples:\n\n"
    "- **Counts:** “How many negative reviews this week?”, “How many reviews mention delivery?”\n"
    "- **Lists:** “Show the top 10 negative reviews this month”, “Positive reviews about price”\n"
    "- **Breakdown:** “What is the sentiment breakdown?”\n"
    "- **Trends:** “Show the weekly trend”\n"
    "- **Products:** “Which product has the most complaints?”, “Average rating by product”\n"
    "- **Reasons:** “What are the top reasons for dissatisfaction?”, “What should we improve first?”\n"
    "- **Compare:** “Compare this week with last week”\n"
    "- **Follow-ups:** after a question, say “and for September?”, “what about the desk lamp?” or "
    "“after fixing delivery, what should we improve first?”\n"
    "- **Try a text:** paste a review and ask “what is the sentiment of this review?”\n"
    "- **Add data:** attach a CSV with the attachment button, or say “add a review”.\n"
    "- **Delete:** “delete review 12” or “delete reviews 3, 5 and 8” (the ID is the first column in the Data view), "
    "or “undo” for the last review you added. You can also tick rows in the Data view table."
)

_SUGGESTIONS = (
    "Try a count (“How many negative reviews this month?”), a list (“Show the top 5 negative reviews”), "
    "a breakdown, a trend, a product ranking (“Which product has the most complaints?”), "
    "or reasons (“What are customers unhappy about?”). Type **help** for more examples."
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def display_row(row: dict[str, Any]) -> dict[str, Any]:
    """Only the fields that may be shown in chat (never the `extra` columns)."""
    label = row.get("sentiment_label")
    return {
        "id": int(row["id"]),
        "text": row["review_text"],
        "product": row.get("product") or "",
        "sentiment": label,
        "confidence": row.get(f"score_{label}") if label else None,
        "date": str(row.get("created_at") or "")[:10],
    }


def _reply(blocks: list[dict], note: str, mode: str | None = None) -> dict[str, Any]:
    return {"blocks": blocks + [{"type": "route", "text": note}], "mode": mode}


def _text(text: str) -> dict[str, str]:
    return {"type": "text", "text": text}


def _actions(label: str, q: Query) -> dict[str, Any]:
    return {"type": "actions", "buttons": [{"label": label, "filters": q.ui_filters()}]}


def _n(count: int, word: str) -> str:
    return f"{count:,} {word}" + ("" if count == 1 else "s")


def _understood(q: Query, how: str) -> str:
    who = " · interpreted by AI planner (validated)" if q.source == "ai" else ""
    return f"{how} · understood as: {q.intent} · {q.scope()}{who}"


def _nothing_found(q: Query, sentiment: bool = True) -> str:
    text = f"I found no {q.scope(sentiment)}."
    if q.search_terms:
        text += " I searched the review text and product names for those words, so a different spelling may still exist."
    return text


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
ACTION_INTENTS = {"help", "add_review", "save_review", "delete_review", "refuse", "analyze_text", "import_help", "review_prompt"}

# Why the quota guard said no -> a short phrase for the notice (the detail text comes from QuotaGuard.try_acquire)
_QUOTA_WORDS = {
    "daily_limit": " (daily limit reached)",
    "daily_provider_quota": " (daily limit reached)",
    "session_limit": " (this chat has used its AI questions)",
    "minute_limit": " (too many requests this minute, try again shortly)",
    "provider_rate_limit": " (Google's rate limit was hit, try again in a minute)",
}


def route(
    question: str,
    db_path: str | Path | None = None,
    today: date | None = None,
    summarizer: Summarizer | None = None,
    analyzer: Any = None,
    planner: Any = None,
    session_id: str = "anon",
    planner_mode: str | None = None,
    previous: Query | None = None,
) -> dict[str, Any]:
    today = today or date.today()
    if not (question or "").strip():
        return _reply([_text("Ask me something about your feedback.")], "No input")

    products = [p for p in database.get_products(db_path) if p]
    parsed = chat_parser.parse_many(question, products, today)
    single = len(parsed) == 1
    rule_qs = [q for _, q in parsed]
    mode = (planner_mode or config.PLANNER_MODE) if planner is not None else "rules"
    q, notice, context = rule_qs[0], None, None
    if single:
        # A follow-up ("and for September?", "after fixing delivery, what next?") borrows what it leaves out from the
        # previous question. The rules do it only on an explicit cue; the AI planner is also shown the previous plan
        # for short questions the rules could not read.
        context = previous if (previous is not None and q.intent not in ACTION_INTENTS and chat_parser.may_refer_back(question, q)) else None
        if context is not None and chat_parser.is_follow_up(question):
            q = chat_parser.inherit_context(q, context, question)
        unsure = not q.confident
    else:
        unsure = any(not x.confident for x in rule_qs)
    wants_ai = (planner is not None and not any(x.intent in ACTION_INTENTS for x in rule_qs)
                and (mode == "llm_first" or (mode == "hybrid" and unsure)))
    if wants_ai and database.count_reviews(db_path) > 0:
        outcome = planner.plan(question, products, today, session_id, database.get_date_range(db_path), previous=context)
        if len(outcome.queries) > 1:  # the planner found several separate questions in the message
            return _answer_plans(outcome, question, db_path, today, summarizer, analyzer)
        if single:
            if outcome.clarification and (outcome.query is None or outcome.query.intent == "unknown"):
                return _reply([_text(outcome.clarification)], "AI planner asked for clarification · nothing was searched")
            if outcome.query is not None and outcome.query.intent != "unknown":
                q = outcome.query
                if context is not None and chat_parser.is_follow_up(question):
                    q.follow_up, q.kept_from_context = True, chat_parser.describe_kept(q, context)
            elif outcome.query is not None and q.intent == "unknown":
                q = outcome.query  # both agree it is not a question about the reviews
        if outcome.query is None and outcome.status in ("quota", "error"):
            why = _QUOTA_WORDS.get(outcome.detail, "") if outcome.status == "quota" else ""
            notice = ("_The AI question helper is unavailable right now" + why
                      + ", so I used the built-in rules, which may misread unusual wording._")

    if single:
        result = _answer(q, question, db_path, today, summarizer, analyzer)
        if q.kept_from_context:
            result["blocks"].insert(len(result["blocks"]) - 1, _text(
                "_Following on from your previous question, I kept: " + ", ".join(q.kept_from_context)
                + ". Say “overall” or “all products” to drop that._"))
        if notice:
            result["blocks"].insert(len(result["blocks"]) - 1, _text(notice))
        if q.intent in chat_parser.READ_INTENTS:
            result["query"] = q  # the caller keeps it as the context for the next question
        return result

    blocks: list[dict] = []
    for i, (part, pq) in enumerate(parsed, 1):
        blocks.append(_text(f"**{i}. {part[:1].upper() + part[1:]}?**"))
        blocks.extend(_answer(pq, part, db_path, today, summarizer, analyzer)["blocks"])
    if notice:
        blocks.append(_text(notice))
    last = parsed[-1][1]
    return {"blocks": blocks, "mode": None, **({"query": last} if last.intent in chat_parser.READ_INTENTS else {})}


def _answer_plans(outcome: Any, question: str, db_path, today: date, summarizer: Summarizer | None, analyzer: Any) -> dict[str, Any]:
    """One answer per question when the AI planner split the message into several plans."""
    blocks: list[dict] = []
    for i, (pq, clar, title) in enumerate(zip(outcome.queries, outcome.clarifications, outcome.titles), 1):
        label = title or f"Question {i}"
        blocks.append(_text(f"**{i}. {label[:1].upper() + label[1:]}{'' if label.endswith('?') else '?'}**"))
        if pq.intent == "unknown":
            blocks.append(_text(clar or "I'm not sure how to answer that one from your reviews, so I skipped it."))
            blocks.append({"type": "route", "text": "AI planner · not a question about the reviews"})
        else:
            blocks.extend(_answer(pq, title or question, db_path, today, summarizer, analyzer)["blocks"])
    if outcome.dropped:
        blocks.append(_text(f"_I couldn't make sense of {_n(outcome.dropped, 'part')} of your message, so I answered the rest._"))
    last = outcome.queries[-1]
    return {"blocks": blocks, "mode": None, **({"query": last} if last.intent in chat_parser.READ_INTENTS else {})}


def _answer(q: Query, question: str, db_path, today: date, summarizer: Summarizer | None, analyzer: Any) -> dict[str, Any]:
    if q.intent == "help":
        return _reply([_text(HELP_TEXT)], "Built-in help")
    if q.intent == "refuse":
        return _reply(
            [_text("I only answer questions about your reviews, so I can't follow instructions hidden in a question, run SQL, "
                   "or change data in bulk. To remove specific reviews, say `delete review 12` (IDs are in the Data view) or tick rows there.")],
            "Declined · not a question about the reviews",
        )
    if q.intent == "delete_review":
        return _delete_review(q, db_path)
    if q.intent == "review_prompt":
        r = _reply(
            [_text("That reads like a review rather than a question, so I didn't search the data for it.\n\n"
                   "> " + (q.text or "")[:300] + "\n\nWhat should I do with it? Reply **analyze** to see its sentiment "
                   "(nothing is saved), **save** to add it to the database, or **cancel**.")],
            "Waiting for your choice · analyze or save",
            mode="review_choice",
        )
        r["review_text"] = q.text
        return r
    if q.intent in ("add_review", "save_review"):
        return _reply(
            [_text("Sure. Type or paste the review as your next message and I will save it and show its sentiment. "
                   "Say `cancel` to stop.\n\nTo tag it, add `Product: Desk Lamp` and `Rating: 4` after the review. "
                   "Without a product it is still saved, but product questions will skip it.\n\n"
                   "You can also do it in one message: *the review text* `Product: Desk Lamp Rating: 4` then `add this review`.")],
            "Action · add review (waiting for text)",
            mode="add_review",
        )
    if q.intent == "analyze_text":
        return _analyze_text(q.text, analyzer)
    if q.intent == "import_help":
        return _reply(
            [_text("Use the attachment button in the message box to attach a CSV. I will detect the review column, "
                   "show you a preview, and ask you to confirm before importing anything.")],
            "Built-in help",
        )
    if database.count_reviews(db_path) == 0:
        return _reply(
            [_text("There are no reviews yet. Attach a CSV with the attachment button, or say “add a review”.")],
            "No data loaded",
        )

    handlers = {
        "count": _count, "breakdown": _breakdown, "trend": _trend, "list": _list,
        "ranking": _ranking, "average": _average, "compare": _compare_any, "peak": _peak, "mismatch": _mismatch,
    }
    if q.intent in handlers or q.intent == "reasons":
        unknown = _unknown_terms(q, db_path)
        if unknown:
            return unknown
    if q.intent == "list" and q.order is None:
        q.order = chat_parser.detect_order(question)  # also for AI plans, which do not carry an order
    if q.intent in handlers:
        result = _with_week_hint(handlers[q.intent](q, db_path, today), q, db_path, today)
        return _with_thin_data_hint(result, q, db_path, today)
    if q.intent == "reasons":
        result = _reasons(q, db_path, summarizer, recommend="improve" in question.lower() or "should we" in question.lower(),
                          question=question)
        return _with_thin_data_hint(_with_week_hint(result, q, db_path, today), q, db_path, today)

    understood = q.scope() if (q.sentiment or q.topic or q.product or q.search_terms or q.period) else None
    text = "I'm not sure what you'd like to know about the feedback, and I would rather say so than guess. "
    if understood:
        text += f"I picked up: {understood}, but not what to do with it. "
    return _reply([_text(text + _SUGGESTIONS)], "No matching intent · declined instead of guessing")


def _unknown_terms(q: Query, db_path) -> dict[str, Any] | None:
    """Stop instead of quietly answering for all reviews when a named product or word matches nothing at all."""
    if not q.search_terms:
        return None
    if analytics.overview(db_path, search=[chat_parser.stem(t) for t in q.search_terms])["total"] > 0:
        return None
    words = " or ".join(f"“{t}”" for t in q.search_terms)
    products = [p for p in database.get_products(db_path) if p]
    text = (f"I have no reviews that mention {words} (I searched the review text and product names), "
            "so I won't answer for all of your reviews instead.")
    if products:
        text += " Products in your data: " + ", ".join(products[:8]) + ("…" if len(products) > 8 else "") + "."
    return _reply([_text(text)], f"Abstained · nothing in the data matches {words}")


THIN_DATA = 10  # fewer matching reviews than this and percentages are too rough to trust
_THIN_INTENTS = ("count", "breakdown", "average", "reasons", "mismatch")


def _with_thin_data_hint(result: dict[str, Any], q: Query, db_path, today: date) -> dict[str, Any]:
    """A period with only a handful of reviews ("this month" on the 1st) gives shaky percentages: say so, and say
    where the data really is, instead of presenting 1 of 2 as "50%"."""
    if q.intent not in _THIN_INTENTS or not q.period:
        return result
    total = analytics.overview(db_path, **q.filters(sentiment=False))["total"]
    if not 0 < total < THIN_DATA:
        return result  # nothing found has its own message; enough reviews needs no warning
    start = today - timedelta(days=29)
    recent = analytics.overview(db_path, date_from=start.isoformat(), date_to=today.isoformat(), **q.filters(period=False, sentiment=False))["total"]
    newest = database.get_date_range(db_path)[1]
    note = f"_Only {_n(total, 'review')} match this time period, so treat these figures as rough."
    if recent > total:
        note += f" The last 30 days ({_fmt_day(start)} – {_fmt_day(today)}) have {recent}; ask about those for a steadier picture."
    if newest and newest < today.isoformat():
        note += f" Your newest review is dated {newest}."
    result["blocks"].insert(len(result["blocks"]) - 1, _text(note + "_"))
    return result


def _fmt_day(d: date) -> str:
    return f"{d:%b} {d.day}"


def _with_week_hint(result: dict[str, Any], q: Query, db_path, today: date) -> dict[str, Any]:
    """"This week" on a Monday covers one day. When that finds nothing, also show the last 7 days."""
    if not (q.period and q.period.label.startswith("this week")):
        return result
    days_so_far = (today - date.fromisoformat(q.period.date_from)).days + 1
    first = result["blocks"][0] if result["blocks"] else None
    if days_so_far > 3 or not first or first.get("type") != "text":
        return result
    if not any(w in first["text"].lower() for w in ("found no", "no reviews", "are no ", "enough evidence", "**0 ")):
        return result
    week = analytics.overview(db_path, date_from=(today - timedelta(days=6)).isoformat(), date_to=today.isoformat(),
                              **q.filters(period=False))
    first["text"] += (f"\n\n_This week has only {_n(days_so_far, 'day')} so far (it starts on Monday). "
                      f"In the last 7 days there {'was' if week['total'] == 1 else 'were'} {_n(week['total'], 'matching review')}._")
    return result


def _delete_review(q: Query, db_path) -> dict[str, Any]:
    """Delete the reviews named by id ("delete review 12", "delete reviews 3, 5 and 8") or the last one added in chat."""
    ids = q.review_ids or []
    if q.delete_last or not ids:
        last = database.latest_review_id("chat", db_path)
        if last is None:
            return _reply([_text("You haven't added any review in this chat yet, so there is nothing to undo. "
                                 "To delete a specific review, use its ID (the first column of the Data view), "
                                 "for example `delete review 12`.")], "Action · delete review (nothing deleted)")
        ids = [last]
    found, missing = [], []
    for review_id in dict.fromkeys(ids):
        row = database.get_review(review_id, db_path) if review_id > 0 else None
        if row is None:
            missing.append(review_id)
        else:
            found.append((review_id, row))
    missing_line = ("No review has the ID " + ", ".join(str(i) for i in missing)
                    + ". IDs start at 1 and are shown in the first column of the Data view.") if missing else ""
    if not found:
        return _reply([_text(missing_line)], "Action · delete review (nothing deleted)")
    preview = "\n".join(
        f"- **ID {i}**" + (f" · {row['product']}" if row.get("product") else "") + f": {(row.get('review_text') or '')[:120]}"
        for i, row in found)
    text = (f"This will permanently delete {_n(len(found), 'review')}:\n{preview}\n\n"
            + (missing_line + "\n\n" if missing_line else "")
            + "Reply **yes** to delete, or **cancel** to keep " + ("it." if len(found) == 1 else "them."))
    result = _reply([_text(text)], "Waiting for your confirmation · nothing deleted yet", mode="delete_confirm")
    result["delete_ids"] = [i for i, _ in found]
    return result


_YES = ("yes", "y", "yes delete", "confirm", "delete", "delete it", "delete them", "do it", "ok", "okay", "sure")
_NO = ("no", "n", "cancel", "stop", "keep", "keep it", "keep them", "don't", "dont", "never mind", "nevermind")


def parse_confirmation(text: str) -> str | None:
    """"yes"/"no" for a reply to a confirmation question, or None when the message is something else."""
    t = (text or "").lower().strip(" .!")
    if t in _YES:
        return "yes"
    if t in _NO or t.startswith("cancel"):
        return "no"
    return None


def confirm_delete(ids: list[int], db_path) -> dict[str, Any]:
    """Second step of "delete review N": the owner said yes, so delete what was previewed."""
    deleted = []
    for review_id in dict.fromkeys(ids):
        row = database.get_review(review_id, db_path) if review_id > 0 else None
        if row is not None:
            database.delete_review(review_id, db_path)
            deleted.append((review_id, (row.get("review_text") or "")[:120]))
    if not deleted:
        return _reply([_text("Those reviews were already gone, so nothing was deleted.")], "Action · delete review (nothing deleted)")
    lines = f"Deleted {_n(len(deleted), 'review')}:\n" + "\n".join(f"- **ID {i}**: {t}" for i, t in deleted)
    return _reply([_text(lines)], "Action · delete review")


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------
def _analyze_text(text: str | None, analyzer: Any) -> dict[str, Any]:
    if not text:
        return _reply([_text("Paste the review text in your message and I will show its sentiment.")], "Sentiment service · no text")
    if analyzer is None:
        return _reply([_text("The sentiment model isn't available right now, so I can't analyse that text.")], "Sentiment service unavailable")
    result = analyzer.analyze(text)
    if not result.ok:
        return _reply([_text(f"I couldn't analyse that text: {result.error}")], "Sentiment service error")
    found_topics = [t for t in topics_service.tag_topics(text) if t != "other"]
    blocks: list[dict] = [
        _text(f"Here is the sentiment of that text. Nothing was saved.\n\n> {text[:300]}{'…' if len(text) > 300 else ''}"
              + (f"\n\nTopics I can see: **{', '.join(found_topics)}**." if found_topics else "")),
        {"type": "sentiment", "label": result.label, "scores": result.scores},
    ]
    if "low_confidence" in result.flags:
        blocks.append({"type": "badge", "kind": "rule", "text": "The model is not very sure about this one."})
    return _reply(blocks, "Local sentiment model · no database write")


def _count(q: Query, db_path, today: date) -> dict[str, Any]:
    base = analytics.overview(db_path, **q.filters(sentiment=False))
    if base["total"] == 0:
        return _reply([_text(_nothing_found(q, sentiment=False))], _understood(q, "SQL template · overview"))
    n = base[q.sentiment] if q.sentiment else base["total"]
    analysed = base["positive"] + base["neutral"] + base["negative"]
    scope = q.scope()
    if q.sentiment:
        share = f" ({n / analysed:.0%} of the {base['total']:,} reviews in this scope)" if analysed else ""
        text = f"There {'is' if n == 1 else 'are'} **{_n(n, q.sentiment + ' review')}** {scope.replace(q.sentiment + ' reviews', '', 1).strip()}{share}."
        text = text.replace("  ", " ").replace(" .", ".")
        if base["pending"]:
            text += f" ({base['pending']} review(s) are not analysed yet and are not counted.)"
    else:
        text = f"There {'is' if n == 1 else 'are'} **{_n(n, 'review')}** {scope.replace('reviews', '', 1).strip()}."
    return _reply([_text(text), _actions("Open in Data view", q)], _understood(q, "SQL template · overview · no AI used"))


def _breakdown(q: Query, db_path, today: date) -> dict[str, Any]:
    scope = analytics.overview(db_path, **q.filters(sentiment=False))
    if scope["total"] == 0:
        return _reply([_text(_nothing_found(q, sentiment=False))], _understood(q, "SQL template · overview"))
    counts = {k: scope[k] for k in ("positive", "neutral", "negative")}
    blocks: list[dict] = [
        _text(f"Sentiment across **{_n(scope['total'], 'review')}** {q.scope(sentiment=False).replace('reviews', '', 1).strip()}:"),
        {"type": "breakdown", "counts": counts},
    ]
    if scope["pending"]:
        blocks.append(_text(f"_{scope['pending']} review(s) have not been analysed yet._"))
    if scope["low_confidence"]:
        blocks.append(_text(f"_{scope['low_confidence']} review(s) have a low-confidence label._"))
    blocks.append(_actions("Open in Data view", q))
    return _reply(blocks, _understood(q, "SQL template · overview · no AI used"))


def _trend(q: Query, db_path, today: date) -> dict[str, Any]:
    filters = q.filters(period=False)
    d_from = q.period.date_from if q.period else None
    d_to = q.period.date_to if q.period else None
    if not q.period:
        base = analytics.overview(db_path, **filters)
        if base["total"] == 0:
            return _reply([_text(_nothing_found(q, sentiment=False))], _understood(q, "SQL template · sentiment_trend"))
        d_from, d_to = base["first_day"], base["last_day"]
    span = (date.fromisoformat(d_to) - date.fromisoformat(d_from)).days if d_from and d_to else 0
    gran = q.granularity or analytics.choose_granularity(span)
    note = ""
    if gran == "week" and span <= 7:
        gran, note = "day", "That period is shorter than a full week, so I show one bar per day. "
    rows = analytics.sentiment_trend(gran, db_path, date_from=d_from, date_to=d_to, **filters)
    if not rows:
        return _reply([_text(_nothing_found(q, sentiment=False))], _understood(q, "SQL template · sentiment_trend"))
    rows = analytics.fill_trend(rows, gran, d_from, d_to)
    for r in rows:
        r["label"] = analytics.period_label(r["period"], gran)
    start_word = {"day": "Day", "week": "Week starting (Monday)", "month": "Month"}[gran]
    total = sum(r["total"] for r in rows)
    trend_filters = {**filters, **({"date_from": d_from, "date_to": d_to} if d_from and d_to else {})}
    reading = insights.describe_trend(f"{q.product + ' ' if q.product else ''}reviews", insights.recent_vs_earlier(db_path, trend_filters))
    return _reply(
        [
            _text(f"{note}**What this shows:** {reading}\n\n"
                  f"Reviews by sentiment, one bar per {gran}"
                  + (f", {q.period.label}" if q.period else ", across all your data")
                  + f" ({_n(total, 'review')} in total):"),
            {"type": "chart", "rows": rows, "x_label": start_word},
            _actions("Open in Data view", q),
        ],
        _understood(q, f"SQL template · sentiment_trend({gran}) · no AI used"),
    )


def _list(q: Query, db_path, today: date) -> dict[str, Any]:
    limit = q.limit or DEFAULT_LIMIT
    rows, total = database.fetch_reviews(sort=q.order or "newest", page=1, page_size=limit, db_path=db_path, **q.filters())
    if not rows:
        return _reply([_text(_nothing_found(q))], _understood(q, "SQL template · recent_reviews"))
    more = f" ({total:,} match in total)" if total > len(rows) else ""
    kind = {"worst": "lowest-rated review", "best": "highest-rated review"}.get(q.order or "", "newest review")
    how = ("\n\n_Ordered by lowest star rating first, then by how strongly the model judged the review negative._" if q.order == "worst"
           else "\n\n_Ordered by highest star rating first, then by how strongly the model judged the review positive._" if q.order == "best" else "")
    return _reply(
        [
            _text(f"The {_n(len(rows), kind)} {q.scope().replace('reviews', '', 1).strip()}{more}:{how}"),
            {"type": "reviews", "rows": [display_row(r) for r in rows]},
            _actions("See all in Data view", q),
        ],
        _understood(q, "SQL template · recent_reviews · no AI used"),
    )


def _ranking(q: Query, db_path, today: date) -> dict[str, Any]:
    stats = analytics.by_product(db_path, **q.filters(sentiment=False))
    named = [s for s in stats if s["product"] != "(no product)"]
    without = sum(s["total"] for s in stats if s["product"] == "(no product)")
    if not named:
        return _reply(
            [_text("None of the matching reviews has a product name, so I can't rank products. "
                   "Map a product column when you import a CSV, and this will work.")],
            _understood(q, "SQL template · by_product"),
        )
    metric = q.sentiment or "total"
    named.sort(key=lambda s: (-s[metric], -s["total"], s["product"]))
    top = named[0]
    kind = f"{metric} reviews" if q.sentiment else "reviews"
    if not top[metric]:
        sentence = f"No product has any {kind} in this scope."
    else:
        tied = [s for s in named if s[metric] == top[metric]]

        def share(s):
            return s[metric] / max(s["positive"] + s["neutral"] + s["negative"], 1)

        leader = max(named, key=lambda s: (share(s), s["total"]))
        if len(tied) > 1:
            names = ", ".join(s["product"] for s in tied[:-1]) + f" and {tied[-1]['product']}"
            sentence = f"**{names}** are tied for the most {kind}, with {top[metric]} each."
        else:
            sentence = f"In raw numbers, **{top['product']}** has the most {kind} ({top[metric]})"
            if top["total"] == max(s["total"] for s in named) and q.sentiment:
                sentence += f", but it also has the most reviews overall ({top['total']}), so that is partly just volume"
            sentence += "."
        if q.sentiment and (leader["product"] != top["product"] or len(tied) > 1):
            sentence += (f" Relative to its size, **{leader['product']}** is the weakest: {leader[metric]} of its "
                         f"{leader['total']} reviews ({share(leader):.0%}) are {q.sentiment}.")
        elif q.sentiment:
            sentence += f" It is also the highest relative to its size ({top[metric]} of {top['total']}, {share(top):.0%})."
    table = {
        "type": "table",
        "columns": ["Product", "Reviews", "Positive", "Neutral", "Negative", "Negative %"],
        "rows": [[s["product"], s["total"], s["positive"], s["neutral"], s["negative"], f"{s['negative_pct']:.0f}%"] for s in named[:10]],
    }
    blocks = [_text(sentence), table]
    if any(s["total"] < 5 for s in named):
        blocks.append(_text("_Some products have fewer than 5 reviews, so their shares are rough._"))
    if without:
        blocks.append(_text(f"_{_n(without, 'review')} without a product name are not included._"))
    blocks.append(_actions("Open in Data view", q))
    return _reply(blocks, _understood(q, f"SQL template · by_product(sorted by {metric}) · no AI used"))


def _average(q: Query, db_path, today: date) -> dict[str, Any]:
    if q.group_by == "product":
        stats = [s for s in analytics.by_product(db_path, **q.filters()) if s["product"] != "(no product)" and s["avg_rating"] is not None]
        if not stats:
            return _reply([_text("No product has rated reviews in this scope.")], _understood(q, "SQL template · by_product"))
        stats.sort(key=lambda s: (-s["avg_rating"], -s["rated"], s["product"]))
        reliable = [s for s in stats if s["rated"] >= 3]
        if reliable:
            best = reliable[0]["avg_rating"]
            tied = [s for s in reliable if s["avg_rating"] == best]
            names = ", ".join(s["product"] for s in tied[:-1]) + (" and " if len(tied) > 1 else "") + tied[-1]["product"]
            head = (f"Among products with at least 3 star ratings, **{names}** {'tie for' if len(tied) > 1 else 'has'} the best "
                    f"average rating: {best:.2f} / 5.")
        else:
            head = "No product has 3 or more star ratings yet, so any ranking would be unreliable. Here are the averages anyway:"
        table = {"type": "table", "columns": ["Product", "Average rating", "Rated reviews", "All reviews", "Negative %"],
                 "rows": [[s["product"], f"{s['avg_rating']:.2f} / 5", s["rated"], s["total"], f"{s['negative_pct']:.0f}%"] for s in stats[:10]]}
        blocks = [_text(head), table]
        if any(s["rated"] < s["total"] for s in stats):
            blocks.append(_text("_Only reviews with a star rating are averaged, so the average can disagree with the sentiment of all reviews "
                                "(see the Negative % column)._"))
        blocks.append(_actions("Open in Data view", q))
        return _reply(blocks, _understood(q, "SQL template · by_product · no AI used"))
    o = analytics.overview(db_path, **q.filters())
    if o["total"] == 0:
        return _reply([_text(_nothing_found(q))], _understood(q, "SQL template · overview"))
    if o["avg_rating"] is None:
        return _reply([_text(f"None of the {_n(o['total'], 'review')} in this scope has a star rating.")], _understood(q, "SQL template · overview"))
    return _reply(
        [_text(f"The average rating is **{o['avg_rating']:.2f} out of 5** across the reviews that have a rating, "
               f"{q.scope().replace('reviews', '', 1).strip()}."), _actions("Open in Data view", q)],
        _understood(q, "SQL template · overview(avg_rating) · no AI used"),
    )


def _bare(label: str) -> str:
    return label[3:] if label.startswith("in ") else label


def _compare_any(q: Query, db_path, today: date) -> dict[str, Any]:
    return _compare_products(q, db_path, today) if q.group_by == "product" else _compare(q, db_path, today)


def _compare_products(q: Query, db_path, today: date) -> dict[str, Any]:
    """"Which product caused the increase in positive reviews in October?": per-product change between two periods.
    Reviews show WHAT moved for each product, never WHY, so the wording says "accounts for", not "caused"."""
    cur = q.period or chat_parser.parse_periods("last 30 days", today)[0]
    if q.previous_period:
        previous = (q.previous_period.date_from, q.previous_period.date_to)
        prev_label = q.previous_period.label
    else:
        previous = product_change.previous_window(cur.date_from, cur.date_to)
        prev_label = f"{_fmt_day(date.fromisoformat(previous[0]))} – {_fmt_day(date.fromisoformat(previous[1]))}"
    result = product_change.product_changes(db_path, (cur.date_from, cur.date_to), previous, **q.filters(period=False, sentiment=False))
    before, after = result["overall"]["total"]
    note = _understood(q, "SQL template · by_product x2 · no AI used")
    if after == 0 and before == 0:
        return _reply([_text(f"There are no reviews in either period ({cur.label} and {prev_label}).")], note)
    if before == 0 or after == 0:
        empty = prev_label if before == 0 else cur.label
        return _reply([_text(f"I can't say what changed per product, because there are no reviews in {empty}.")], note)
    focus = q.sentiment if q.sentiment in ("positive", "negative") else None
    lines = product_change.describe(result, focus)
    blocks: list[dict] = [
        _text(f"Comparing **{_bare(cur.label)}** with **{_bare(prev_label)}**:\n\n" + "\n\n".join(lines)),
        {"type": "table", "columns": product_change.TABLE_COLUMNS, "rows": product_change.table_rows(result)},
        _text("_Reviews show what changed for each product, not why it changed. Counts also move when more or fewer reviews arrive, "
              "so check the negative-share column too._"),
    ]
    if any(r["few"] for r in result["rows"]):
        blocks.append(_text("_Products marked “few reviews” have fewer than 5 in one of the periods, so they are not named as drivers._"))
    blocks.append(_actions("Open in Data view", q))
    return _reply(blocks, note)


def _compare(q: Query, db_path, today: date) -> dict[str, Any]:
    filters = q.filters(period=False)
    cur = q.period or chat_parser.parse_periods("last 7 days", today)[0]
    if q.previous_period:
        current = analytics.overview(db_path, date_from=cur.date_from, date_to=cur.date_to, **filters)
        previous = analytics.overview(db_path, date_from=q.previous_period.date_from, date_to=q.previous_period.date_to, **filters)
        prev_label = q.previous_period.label
    else:
        result = analytics.compare_periods(cur.date_from, cur.date_to, db_path, **filters)
        current, previous = result["current"], result["previous"]
        prev_label = f"the {(date.fromisoformat(cur.date_to) - date.fromisoformat(cur.date_from)).days + 1} days before ({result['previous_range'][0]} to {result['previous_range'][1]})"
    if current["total"] == 0 and previous["total"] == 0:
        return _reply([_text("There are no reviews in either period.")], _understood(q, "SQL template · compare"))
    if current["total"] == 0 or previous["total"] == 0:
        empty = cur.label if current["total"] == 0 else prev_label
        text = f"I can't say whether anything changed, because there are no reviews {empty if empty.startswith('in ') else 'in ' + empty}."
    else:
        change = current["negative_pct"] - previous["negative_pct"]
        verb = "rose" if change > 0 else "fell" if change < 0 else "stayed the same"
        text = f"Negative reviews **{verb}**" + (f" by {abs(change):.0f} percentage points" if change else "") + \
            f", from {previous['negative_pct']:.0f}% to {current['negative_pct']:.0f}%."
        if min(current["total"], previous["total"]) < 5:
            text = "Only a rough read, because one period has very few reviews: " + text[0].lower() + text[1:]
    table = {
        "type": "table",
        "columns": ["Period", "Reviews", "Positive %", "Negative %", "Avg rating"],
        "rows": [
            [cur.label, current["total"], f"{current['positive_pct']:.0f}%", f"{current['negative_pct']:.0f}%", current["avg_rating"] if current["avg_rating"] is not None else "n/a"],
            [prev_label, previous["total"], f"{previous['positive_pct']:.0f}%", f"{previous['negative_pct']:.0f}%", previous["avg_rating"] if previous["avg_rating"] is not None else "n/a"],
        ],
    }
    blocks: list[dict] = [_text(text), table]
    if min(current["total"], previous["total"]) < 5:
        blocks.append(_text("_One of the periods has fewer than 5 reviews, so treat this as weak evidence._"))
    blocks.append(_actions("Open in Data view", q))
    return _reply(blocks, _understood(q, "SQL template · overview x2 · no AI used"))


def _peak(q: Query, db_path, today: date) -> dict[str, Any]:
    gran = q.granularity or "month"
    base = analytics.overview(db_path, **q.filters(sentiment=False))
    if base["total"] == 0:
        return _reply([_text(_nothing_found(q, sentiment=False))], _understood(q, "SQL template · sentiment_trend"))
    rows = analytics.sentiment_trend(gran, db_path, **q.filters(sentiment=False))
    metric = q.sentiment or "total"
    pick = min if q.extreme == "min" else max
    target = pick(r[metric] for r in rows)
    winners = [r for r in rows if r[metric] == target]

    def nice(r):
        label = analytics.period_label(r["period"], gran)
        if gran == "month":
            try:
                return datetime.strptime(label, "%Y-%m").strftime("%B %Y")
            except ValueError:
                pass
        return label

    def partial(r):
        """First/last month of the data usually covers only part of the month."""
        if gran != "month":
            return False
        if r is rows[0] and base["first_day"] and int(base["first_day"][8:10]) > 1:
            return True
        return r is rows[-1] and bool(base["last_day"]) and (date.fromisoformat(base["last_day"]) + timedelta(days=1)).month == date.fromisoformat(base["last_day"]).month

    kind = f"{metric} reviews" if q.sentiment else "reviews"
    word = "fewest" if q.extreme == "min" else "most"
    names = " and ".join(nice(r) for r in winners[:3])
    text = (f"**{names}** {'are tied for' if len(winners) > 1 else 'had'} the {word} {kind}"
            f" ({target}{' each' if len(winners) > 1 else ''}).")
    if q.sentiment and len(winners) == 1 and rows:
        w = winners[0]
        if w["total"] == max(r["total"] for r in rows) and len(rows) > 1:
            text += f" It also had the most reviews overall ({w['total']}), so that is partly just volume."
        eligible = [r for r in rows if r["total"] >= 10]
        if eligible:
            best = max(eligible, key=lambda r: (r[metric] / r["total"], r["total"]))
            if best is not w:
                text += (f" As a share of its own reviews, **{nice(best)}** was highest: {best[metric]} of {best['total']} "
                         f"({best[metric] / best['total']:.0%})" + (", though your data covers only part of that month" if partial(best) else "") + ".")
    partials = [nice(r) for r in rows if partial(r)]
    table = {"type": "table", "columns": [gran.capitalize(), "Reviews", "Positive", "Neutral", "Negative", "Negative %"],
             "rows": [[nice(r) + (" (partial)" if partial(r) else ""), r["total"], r["positive"], r["neutral"], r["negative"],
                       f"{r['negative'] / max(r['positive'] + r['neutral'] + r['negative'], 1):.0%}"] for r in rows[-12:]]}
    blocks = [_text(text), table]
    if partials:
        blocks.append(_text(f"_{' and '.join(partials)} {'is' if len(partials) == 1 else 'are'} only partly covered by your data, so {'its' if len(partials) == 1 else 'their'} totals are lower than a full month's._"))
    if len(rows) < 3:
        blocks.append(_text(f"_Your data covers only {_n(len(rows), gran)}, so this comparison is not very meaningful._"))
    blocks.append(_actions("Open in Data view", q))
    return _reply(blocks, _understood(q, f"SQL template · sentiment_trend({gran}) · no AI used"))


def _mismatch(q: Query, db_path, today: date) -> dict[str, Any]:
    stats = [s for s in analytics.by_product(db_path, **q.filters(sentiment=False))
             if s["product"] != "(no product)" and s["avg_rating"] is not None]
    rows, _ = database.fetch_reviews(sort="newest", page=1, page_size=500, db_path=db_path, **q.filters(sentiment=False))
    rated = [r for r in rows if r.get("rating") is not None and r.get("sentiment_label")]
    if not rated:
        return _reply([_text("None of the matching reviews has both a star rating and a sentiment, so I can't compare them.")],
                      _understood(q, "SQL template · rating vs sentiment"))
    clash = [r for r in rated if (r["sentiment_label"] == "positive" and r["rating"] <= 2)
             or (r["sentiment_label"] == "negative" and r["rating"] >= 4)]
    flagged = [s["product"] for s in stats if s["rated"] >= 2 and s["positive_pct"] >= 50 and s["avg_rating"] <= 3]
    head = (f"Among {_n(len(rated), 'review')} with both a rating and a sentiment, **{len(clash)}** clearly disagree "
            "(positive text with 1–2 stars, or negative text with 4–5 stars).")
    if flagged:
        head += f" Products that are mostly positive but average 3 stars or less: **{', '.join(flagged)}**."
    else:
        head += " No product is mostly positive while averaging 3 stars or less."
        close = [s for s in stats if s["rated"] >= 5 and s["positive_pct"] >= 40]
        if close:
            c = min(close, key=lambda s: s["avg_rating"])
            head += f" The closest is **{c['product']}**: {c['positive_pct']:.0f}% positive reviews with a {c['avg_rating']:.1f}-star average."
    table = {"type": "table", "columns": ["Product", "Positive %", "Average rating", "Rated reviews"],
             "rows": [[s["product"], f"{s['positive_pct']:.0f}%", f"{s['avg_rating']:.2f}", s["rated"]] for s in stats[:10]]}
    blocks = [_text(head), table]
    if clash:
        blocks += [_text("Reviews where rating and text disagree (newest first):"),
                   {"type": "reviews", "rows": [display_row(r) for r in clash[:5]]}]
    blocks.append(_actions("Open in Data view", q))
    return _reply(blocks, _understood(q, "SQL template · rating vs sentiment · no AI used"))


def _summary_sample(q: Query, db_path, size: int = 12) -> list[dict[str, Any]]:
    """Retrieval for an AI summary: a small, balanced sample of the matching reviews (newest first within each
    sentiment, taken in turn) so one loud sentiment does not crowd out the others."""
    rows, _ = database.fetch_reviews(sort="newest", page=1, page_size=60, db_path=db_path, **q.filters())
    groups: dict[str, list[dict[str, Any]]] = {"negative": [], "positive": [], "neutral": []}
    for r in rows:
        groups.setdefault(r.get("sentiment_label") or "neutral", []).append(r)
    picked: list[dict[str, Any]] = []
    while len(picked) < size and any(groups.values()):
        for g in groups.values():
            if g and len(picked) < size:
                picked.append(g.pop(0))
    return [display_row(r) for r in sorted(picked, key=lambda r: (str(r.get("created_at")), int(r["id"])), reverse=True)]


def _reasons(q: Query, db_path, summarizer: Summarizer | None, recommend: bool, question: str = "") -> dict[str, Any]:
    focus = q.sentiment or "negative"
    # "Leaving out delivery" here means: do not rank delivery as an issue. Reviews that mention delivery AND something
    # else still count for that something else, so the reviews themselves are not filtered away.
    base_filters = q.filters(sentiment=False, exclude=False)
    scope = analytics.overview(db_path, **base_filters)
    scope_text = q.scope(sentiment=False, exclusions=False).replace("reviews", "", 1).strip() or "overall"
    if scope["total"] == 0:
        return _reply(
            [_text(f"I don't have enough evidence to answer: there are no reviews {scope_text}."
                   + (" I searched the review text and product names for those words." if q.search_terms else ""))],
            _understood(q, "Retrieval found nothing · abstained"),
        )
    n_focus = scope[focus]
    if n_focus == 0:
        newest, _ = database.fetch_reviews(sort="newest", page=1, page_size=3, db_path=db_path, **base_filters)
        text = (f"There are no {focus} reviews among the {_n(scope['total'], 'matching review')} ({scope_text}): "
                f"{scope['positive']} positive, {scope['neutral']} neutral, {scope['negative']} negative.")
        blocks0: list[dict] = [_text(text)]
        if newest:
            blocks0 += [_text("The newest matching reviews:"), {"type": "reviews", "rows": [display_row(r) for r in newest]}]
        return _reply(blocks0, _understood(q, "Retrieval (SQL filters + topic tags) · no AI needed"))

    topics_found = analytics.topic_breakdown(db_path, sentiment=focus, **base_filters)
    ranked = [t for t in topics_found if t["topic"] != "other"] or topics_found
    top = ranked[0] if ranked else None
    if q.topic:  # every review in this scope has that topic by definition, so list the other topics instead
        ranked = [t for t in ranked if t["topic"] != q.topic]
    left_out = {t["topic"]: t[focus] for t in ranked if t["topic"] in q.exclude_topics}
    ranked = [t for t in ranked if t["topic"] not in q.exclude_topics]
    top = ranked[0] if ranked else None
    tied = [t for t in ranked if top and t[focus] == top[focus]]
    where = scope_text.replace(" overall", "").strip()
    where = where if where and where != "overall" else "in your data"
    analysed = max(scope["positive"] + scope["neutral"] + scope["negative"], 1)
    facts_text = (f"**In short:** of the {scope['total']} reviews {where}, {n_focus} ({n_focus / analysed:.0%}, "
                  f"{insights.one_in(n_focus, analysed)}) are {focus}.")
    if ranked:
        items = [f"{t['topic']} ({t[focus]} of {n_focus})" if i == 0 else f"{t['topic']} ({t[focus]})" for i, t in enumerate(ranked[:3])]
        listed = items[0] if len(items) == 1 else ", ".join(items[:-1]) + f" and {items[-1]}"
        lead = "Other issues mentioned in them" if q.topic else ("The issues mentioned most in those negative reviews" if focus == "negative" else f"The things mentioned most in those {focus} reviews")
        facts_text += f" {lead}: {listed}. One review can mention several."
    if q.exclude_topics:
        shown = ", ".join(f"{t} ({left_out.get(t, 0)} of {n_focus})" for t in q.exclude_topics)
        facts_text += f" Leaving out {shown}, as you asked."
        if not ranked:
            facts_text += " Nothing else stands out in these reviews."
    if q.topic or q.product or q.search_terms:
        facts_text += (f" Of those {scope['total']} reviews: {scope['positive']} positive, {scope['neutral']} neutral, "
                       f"{scope['negative']} negative.")
    evidence = analytics.examples(4, db_path, sentiment=focus, **base_filters)
    evidence_rows = [display_row(r) for r in evidence]

    blocks: list[dict] = []
    if scope["total"] < 5:
        blocks.append(_text(f"_Only {_n(scope['total'], 'review')} match{'es' if scope['total'] == 1 else ''}, so treat this as weak evidence._"))

    body = None
    evidence_head, evidence_label = f"{n_focus} {focus} reviews", f"{len(evidence_rows)} newest shown"
    route_note = "Rule-based summary calculated from the database · no AI used"
    if summarizer is not None and chat_parser.wants_summary(question):  # only "summarise / what do customers say" questions
        sample = _summary_sample(q, db_path)
        facts = {"scope": scope_text, "totals": scope, "focus": focus, "topics": ranked, "filters": base_filters,
                 "question": question, "max_sentences": chat_parser.summary_sentences(question),
                 "facts_text": re.sub(r"[*_]", "", facts_text), "sample": sample}
        try:
            body = summarizer(facts, evidence_rows)
        except Exception:  # noqa: BLE001 - any provider problem falls back to rules
            body = None
        if body:
            blocks.append({"type": "badge", "kind": "ai",
                           "text": "AI-written summary of a sample of the matching reviews. The numbers above are calculated; check the cited reviews under Evidence."})
            body = f"{facts_text}\n\n{body}"
            route_note = "Retrieval → language model → checked against the data · answer shows its evidence"
            cited = set(facts.get("cited_ids") or [])
            if cited:
                evidence_rows = [r for r in sample if r["id"] in cited] or evidence_rows
                evidence_head, evidence_label = f"{scope['total']} matching reviews", f"{len(evidence_rows)} cited reviews shown"
        else:
            blocks.append({"type": "badge", "kind": "rule",
                           "text": "The AI summary was unavailable or could not be verified against the data, so this is the calculated summary."})
            route_note = "Rule-based fallback · language model unavailable or its answer failed a check"
    elif summarizer is None and chat_parser.wants_summary(question):
        blocks.append({"type": "badge", "kind": "rule",
                       "text": "AI summaries are switched off here (no Gemini key/model, or AI_SUMMARIES=0), so this is the calculated summary, not a written one."})
        route_note = "Rule-based summary · AI summaries are off"
    if not body:
        body = facts_text
        details = insights.topic_details(db_path, base_filters, focus, [t["topic"] for t in ranked[:3]]) if (ranked and recommend) else {}
        if top and len(tied) > 1:
            names = ", ".join(t["topic"] for t in tied[:-1]) + f" and {tied[-1]['topic']}"
            body += f"\n\n**{names[:1].upper() + names[1:]}** are tied at the top ({top[focus]} each), so there is no single biggest reason by count."
        elif top and not recommend:
            body += f"\n\nThe biggest single issue is **{top['topic']}** ({top[focus]} of {n_focus} {focus} reviews)."
        if top and recommend:
            candidates = tied if len(tied) > 1 else [top]
            order = sorted(candidates, key=lambda t: (not details.get(t["topic"], {}).get("rising", False), -t[focus], t["topic"]))
            pick = order[0]
            d = details.get(pick["topic"], {})
            text = f"**Suggestion:** start with **{pick['topic']}**, which appears in {pick[focus]} of {n_focus} {focus} reviews"
            if len(candidates) > 1 and d.get("rising"):
                text += f" (tied with {', '.join(t['topic'] for t in candidates if t is not pick)}, but it is the one that is getting worse)"
            text += "."
            if d.get("top_product") and d["top_product"][1] >= 2:
                text += f" Most of them are about **{d['top_product'][0]}** ({d['top_product'][1]})."
            if d.get("rising"):
                text += f" {d['recent']} of them came in the last {d['window']}, more than you would expect, so it is getting worse."
            others = [t for t in ranked[:3] if t is not pick]
            if others:
                o = others[0]
                od = details.get(o["topic"], {})
                text += f" Next: **{o['topic']}** ({o[focus]})" + (f", mostly about {od['top_product'][0]}" if od.get("top_product") and od["top_product"][1] >= 2 else "") + "."
            body += "\n\n" + text
    blocks.append(_text(body))
    if evidence_rows:
        blocks.append({
            "type": "evidence",
            "scope": f"{evidence_head} · {scope_text} · {evidence_label}",
            "rows": evidence_rows,
        })
    blocks.append(_actions("Open in Data view", q))
    return _reply(blocks, _understood(q, route_note))
