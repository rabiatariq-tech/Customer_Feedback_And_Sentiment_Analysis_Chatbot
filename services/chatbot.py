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
from services import insights, wording
from services import concepts
from services import owner_questions
from services import issues as issues_service
from services import topics as topics_service
from services.chat_parser import DEFAULT_LIMIT, Query

# summarizer(facts, examples) -> text, or None/raise when unavailable
Summarizer = Callable[[dict[str, Any], list[dict[str, Any]]], "str | None"]

HELP_TEXT = (
    "Ask me anything about your reviews, in your own words. I only use what's in your data. Here are some ideas:\n\n"
    "- **Counts:** “How many negative reviews did we get this week?”, “How many reviews mention delivery?”\n"
    "- **Examples:** “Show me the 10 newest negative reviews”, “Positive reviews about price”\n"
    "- **The big picture:** “What's the sentiment breakdown?”, “Show me the weekly trend”\n"
    "- **Products:** “Which product has the most complaints?”, “What's the average rating by product?”\n"
    "- **Reasons:** “Why are customers unhappy?”, “What should we fix first?”\n"
    "- **Changes:** “Compare this week with last week”, “Which product drove the rise in positive reviews?”\n"
    "- **Follow-ups:** after a question, just say “and for September?”, “what about the desk lamp?” or "
    "“what should we fix first if we leave delivery out?”\n"
    "- **Try a text:** paste a review and ask “what's the sentiment of this review?”\n"
    "- **Add data:** attach a CSV with the attachment button, or say “add a review”.\n"
    "- **Delete:** “delete review 12” or “delete reviews 3, 5 and 8” (the ID is the first column in the Data view), "
    "or “undo” for the last review you added. You can also tick rows in the Data view table."
)

_SUGGESTIONS = (
    "You could try “How many negative reviews did we get this month?”, “Which product has the most complaints?” "
    "or “What should we fix first?”. Type **help** for more ideas."
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


def _coverage(db_path) -> str:
    """"Your reviews run from Jul 28 to Oct 25." (empty when the dates are unknown)."""
    first, last = database.get_date_range(db_path)
    if not first or not last:
        return ""
    return f" Your reviews run from {_fmt_day(date.fromisoformat(first[:10]))} to {_fmt_day(date.fromisoformat(last[:10]))}."


def _nothing_found(q: Query, sentiment: bool = True, db_path=None, today: date | None = None) -> str:
    text = f"I couldn't find any {wording.phrase(q, sentiment=sentiment, today=today)}."
    if q.search_terms:
        text += " I searched the review text and product names for those words, so a different spelling might still be in there."
    if q.period and db_path is not None:
        text += _coverage(db_path)
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
    "provider_unreachable": " (Google did not answer in time; trying again in a minute)",
}

_TRAILING_WHY = re.compile(r"^(?P<main>.+?)[\s,;]+(?:and\s+)?why(?:\s+(?:is\s+that|so|is\s+it|are\s+they|do\s+they))?\s*[?.!]*\s*$", re.I)


def _ranking_with_why(question: str, products: list[str], today: date, db_path, summarizer, analyzer, planner, session_id, planner_mode) -> dict[str, Any] | None:
    """"Which product gets the most complaints and why?": answer the ranking, then explain THAT product.

    Without this the AI sometimes dropped the "why" and, when it kept it, explained all reviews instead of the product named
    in the first answer. Only ranking and rating-by-product questions are handled here; other questions follow the normal path."""
    m = _TRAILING_WHY.match((question or "").strip())
    if not m:
        return None
    main = m.group("main").strip(" ,;")
    mq = chat_parser.parse(main, products, today)
    if mq.group_by != "product" or mq.intent not in ("ranking", "average") or not database.get_products(db_path):
        return None
    sentiment = (("negative" if mq.extreme == "min" or chat_parser.wants_lowest(main) else "positive") if mq.intent == "average"
                 else mq.sentiment)
    if sentiment not in ("negative", "positive"):
        return None
    first = route(main, db_path, today, summarizer, analyzer, planner, session_id, planner_mode)
    stats = [s for s in analytics.by_product(db_path, **mq.filters(sentiment=False)) if s["product"] != "(no product)"]
    if mq.intent == "average":
        rated = [s for s in stats if s["avg_rating"] is not None and s["rated"] >= 3]
        pick = (min if sentiment == "negative" else max)(rated, key=lambda s: s["avg_rating"]) if rated else None
    else:
        pick = max(stats, key=lambda s: (s[sentiment], s["total"])) if stats else None
    if pick is None or not pick[sentiment]:
        return first
    wq = Query(intent="reasons", sentiment=sentiment, product=pick["product"], period=mq.period, topic=mq.topic, source=mq.source)
    heading = f"**2. Why is {pick['product']} at the top?**" if mq.intent == "ranking" else f"**2. Why is {pick['product']} {'rated lowest' if sentiment == 'negative' else 'rated best'}?**"
    blocks = [_text(f"**1. {main[:1].upper() + main[1:]}?**"), *first["blocks"], _text(heading),
              *_answer(wq, f"why {pick['product']}", db_path, today, summarizer, analyzer)["blocks"]]
    return {"blocks": blocks, "mode": None, "query": wq}



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
        return _reply([_text("Go ahead and ask me something about your reviews.")], "No input")

    products = [p for p in database.get_products(db_path) if p]
    with_why = _ranking_with_why(question, products, today, db_path, summarizer, analyzer, planner, session_id, planner_mode)
    if with_why is not None:
        return with_why
    special = owner_questions.detect(question, products, today) if database.count_reviews(db_path) > 0 else None
    if special is not None:
        return _owner_question(special, question, db_path, today, summarizer, analyzer)
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
        if not chat_parser.mentions_time(question, today) and not (context is not None and chat_parser.is_follow_up(question)):
            for planned in outcome.queries or ([outcome.query] if outcome.query is not None else []):
                if planned is not None and planned.intent != "compare":
                    planned.period = planned.previous_period = None  # the AI added dates nobody asked for
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
            if not why and outcome.status in ("error", "invalid") and outcome.detail:
                code = re.match(r"HTTP \d{3}|[A-Za-z]*(?:Error|Timeout)\w*|unexpected response shape|bad response", outcome.detail)
                short = outcome.detail if len(outcome.detail) <= 60 and not re.search(r"[<>{}]", outcome.detail) else ""
                why = (f" ({code.group(0)})" if code else f" ({short})" if short and outcome.status == "error"
                       else " (the AI's answer could not be used)" if outcome.status == "invalid" else "")
            notice = ("_The AI question helper isn't available right now" + why
                      + ", so I used my built-in rules instead, which can misread unusual wording._")

    if single:
        result = _answer(q, question, db_path, today, summarizer, analyzer)
        if q.kept_from_context:
            result["blocks"].insert(len(result["blocks"]) - 1, _text(
                "_I carried over " + wording.join_names(list(q.kept_from_context))
                + " from your last question. Say “overall” or “all products” if you don't want that._"))
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
        label = title or wording.phrase(pq, today=today).strip().capitalize() or f"Question {i}"
        label = re.sub(r"^(?:and|also|then)\s+", "", label, flags=re.I)
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
            [_text("I can only answer questions about your reviews, so I can't follow instructions hidden in a message, run SQL "
                   "or change data in bulk. If you want specific reviews gone, say `delete review 12` (the ID is in the Data view) or tick rows there.")],
            "Declined · not a question about the reviews",
        )
    if q.intent == "delete_review":
        return _delete_review(q, db_path)
    if q.intent == "review_prompt":
        r = _reply(
            [_text("That looks like a review rather than a question, so I haven't searched your data for it.\n\n"
                   "> " + (q.text or "")[:300] + "\n\nWhat would you like me to do with it? Reply **analyze** to see its sentiment "
                   "(nothing gets saved), **save** to add it to your data, or **cancel**.")],
            "Waiting for your choice · analyze or save",
            mode="review_choice",
        )
        r["review_text"] = q.text
        return r
    if q.intent in ("add_review", "save_review"):
        return _reply(
            [_text("Sure, paste the review as your next message and I'll save it and show you its sentiment (say `cancel` to stop).\n\n"
                   "If you'd like it tagged, add `Product: Desk Lamp` and `Rating: 4` after the review. Without a product I'll still save it, "
                   "but product questions will skip it.\n\n"
                   "You can also do it all in one message: *the review text* `Product: Desk Lamp Rating: 4` then `add this review`.")],
            "Action · add review (waiting for text)",
            mode="add_review",
        )
    if q.intent == "analyze_text":
        return _analyze_text(q.text, analyzer)
    if q.intent == "import_help":
        return _reply(
            [_text("Use the attachment button in the message box to attach a CSV. I'll find the review column, "
                   "show you a preview and ask you to confirm before anything is imported.")],
            "Built-in help",
        )
    if chat_parser.asks_about_customers(question):
        return _reply(
            [_text("Your reviews don't say who wrote them, so I can't tell you which customers are behind them. "
                   "What I can tell you is which **product** gets the most complaints, or what the complaints are about. Just ask.")],
            "Declined · the data has no customer identities",
        )
    if database.count_reviews(db_path) == 0:
        return _reply(
            [_text("There aren't any reviews yet. Attach a CSV with the attachment button, or say “add a review”.")],
            "No data loaded",
        )

    named = chat_parser.compares_items(question, [p for p in database.get_products(db_path) if p])
    if named and q.intent not in ("peak", "mismatch"):
        return _compare_named_products(q, named, db_path, today)
    # Fix-ups that apply to rule-based and AI plans alike
    if chat_parser.is_opinion(question) and q.intent in ("count", "breakdown", "list", "ranking", "average", "reasons"):
        q.intent, q.sentiment = "reasons", None  # "are customers happy with delivery?" asks how they feel, not for a count
    if q.intent == "average" and q.group_by is None and re.search(
            r"\bwhich\s+(?:\w+\s+)?products?\b|\bproducts?\b[^?.]{0,25}\b(?:highest|lowest|best|worst|top|bottom)\b|\b(?:highest|lowest|best|worst)[- ]rated\b|\bby product\b",
            question.lower()):
        q.group_by = "product"  # "which product has the lowest average rating" wants a product, not one overall number
    if q.intent == "peak" and not re.search(r"\b(day|days|daily|week|weekly|weeks|month|monthly|months|date)\b", question.lower()):
        q.granularity = None  # "when did we get the most ..." with no unit: months, then the busiest week inside
    if q.intent == "average" and q.group_by == "product":
        if chat_parser.wants_lowest(question):
            q.extreme = "min"
        if not chat_parser.mentions_rating(question):
            q.intent, q.sentiment = "ranking", q.sentiment or "positive"  # "which product do customers love the most" is not about stars
        elif not re.search(r"positiv|negativ|neutral", question.lower()):
            q.sentiment = None  # a sentiment filter would hide the other reviews and zero the Negative % column

    if q.intent == "compare" and q.group_by == "product" and q.direction is None:
        ql_all = question.lower()
        q.direction = ("better" if re.search(r"improv|better|\bbest\b|progress|gained|turned around", ql_all)
                       else "worse" if re.search(r"worse|declin|deteriorat|\bdrop|slip|\bworst\b|got worse|fell", ql_all) else None)

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
    text = "I'm not sure what you'd like to know, and I'd rather say so than guess. "
    if understood:
        text += f"I picked up “{wording.phrase(q, today=today)}”, but not what you want to do with it. "
    return _reply([_text(text + _SUGGESTIONS)], "No matching intent · declined instead of guessing")


def _unknown_terms(q: Query, db_path) -> dict[str, Any] | None:
    """Stop instead of quietly answering for all reviews when a named product or word matches nothing at all."""
    if not q.search_terms:
        return None
    if analytics.overview(db_path, search=concepts.expand([chat_parser.stem(t) for t in q.search_terms]))["total"] > 0:
        return None
    words = " or ".join(f"“{t}”" for t in q.search_terms)
    products = [p for p in database.get_products(db_path) if p]
    text = (f"I couldn't find any reviews mentioning {words} (I searched the review text and product names), "
            "so I won't answer for all of your reviews instead.")
    if products:
        text += " The products in your data are " + wording.join_names(products[:8]) + ("…" if len(products) > 8 else "") + "."
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
    note = f"_Heads-up: only {_n(total, 'review')} match this time period, so treat the percentages as rough."
    if recent > total:
        note += f" The last 30 days ({_fmt_day(start)} – {_fmt_day(today)}) have {recent}, which would give a steadier picture."
    if newest and newest < today.isoformat():
        note += f" Your newest review is dated {_fmt_day(date.fromisoformat(newest[:10]))}."
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
    if not any(w in first["text"].lower() for w in ("couldn't find any", "no reviews", "aren't any", "are no ", "enough evidence", "**0 ")):
        return result
    week = analytics.overview(db_path, date_from=(today - timedelta(days=6)).isoformat(), date_to=today.isoformat(),
                              **q.filters(period=False))
    first["text"] += (f"\n\n_We're only {_n(days_so_far, 'day')} into the week (weeks start on Monday). "
                      f"In the last 7 days there {'was' if week['total'] == 1 else 'were'} {_n(week['total'], 'matching review')}._")
    return result


def _delete_review(q: Query, db_path) -> dict[str, Any]:
    """Delete the reviews named by id ("delete review 12", "delete reviews 3, 5 and 8") or the last one added in chat."""
    ids = q.review_ids or []
    if q.delete_last or not ids:
        last = database.latest_review_id("chat", db_path)
        if last is None:
            return _reply([_text("You haven't added any reviews in this chat yet, so there's nothing to undo. "
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
    missing_line = ("There's no review with the ID " + ", ".join(str(i) for i in missing)
                    + ". IDs start at 1 and are shown in the first column of the Data view.") if missing else ""
    if not found:
        return _reply([_text(missing_line)], "Action · delete review (nothing deleted)")
    preview = "\n".join(
        f"- **ID {i}**" + (f" · {row['product']}" if row.get("product") else "") + f": {(row.get('review_text') or '')[:120]}"
        for i, row in found)
    text = (f"This will permanently delete {_n(len(found), 'review')}:\n{preview}\n\n"
            + (missing_line + "\n\n" if missing_line else "")
            + "Reply **yes** to go ahead, or **cancel** to keep " + ("it." if len(found) == 1 else "them."))
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
    lines = f"Done, I've deleted {_n(len(deleted), 'review')}:\n" + "\n".join(f"- **ID {i}**: {t}" for i, t in deleted)
    return _reply([_text(lines)], "Action · delete review")


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------
def _analyze_text(text: str | None, analyzer: Any) -> dict[str, Any]:
    if not text:
        return _reply([_text("Paste the review text into your message and I'll show you its sentiment.")], "Sentiment service · no text")
    if analyzer is None:
        return _reply([_text("The sentiment model isn't available right now, so I can't analyse that text yet.")], "Sentiment service unavailable")
    result = analyzer.analyze(text)
    if not result.ok:
        return _reply([_text(f"I couldn't analyse that text: {result.error}")], "Sentiment service error")
    found_topics = [t for t in topics_service.tag_topics(text) if t != "other"]
    blocks: list[dict] = [
        _text(f"Here's the sentiment of that text (nothing was saved):\n\n> {text[:300]}{'…' if len(text) > 300 else ''}"
              + (f"\n\nTopics I can see: **{wording.join_names(found_topics)}**." if found_topics else "")),
        {"type": "sentiment", "label": result.label, "scores": result.scores},
    ]
    if "low_confidence" in result.flags:
        blocks.append({"type": "badge", "kind": "rule", "text": "The model isn't very sure about this one."})
    return _reply(blocks, "Local sentiment model · no database write")


def _count(q: Query, db_path, today: date) -> dict[str, Any]:
    base = analytics.overview(db_path, **q.filters(sentiment=False))
    if base["total"] == 0:
        return _reply([_text(_nothing_found(q, sentiment=False, db_path=db_path, today=today))], _understood(q, "SQL template · overview"))
    n = base[q.sentiment] if q.sentiment else base["total"]
    analysed = base["positive"] + base["neutral"] + base["negative"]
    tail = wording.rest(q, today=today)  # " about price in October"
    verb = "is" if n == 1 else "are"
    if q.sentiment:
        text = f"There {verb} **{_n(n, q.sentiment + ' review')}**{tail}"
        if analysed:
            text += f". That's {n / analysed:.0%} of all {base['total']:,} reviews{tail or ' you have'}."
        else:
            text += "."
        if base["pending"]:
            text += f" ({_n(base['pending'], 'review')} haven't been analysed yet, so they aren't counted.)"
    else:
        text = f"There {verb} **{_n(n, 'review')}**{tail or ' in total'}."
    return _reply([_text(text), _actions("Open in Data view", q)], _understood(q, "SQL template · overview · no AI used"))


def _breakdown(q: Query, db_path, today: date) -> dict[str, Any]:
    scope = analytics.overview(db_path, **q.filters(sentiment=False))
    if scope["total"] == 0:
        return _reply([_text(_nothing_found(q, sentiment=False, db_path=db_path, today=today))], _understood(q, "SQL template · overview"))
    counts = {k: scope[k] for k in ("positive", "neutral", "negative")}
    blocks: list[dict] = [
        _text(f"Here's how the **{_n(scope['total'], 'review')}**{wording.rest(q, today=today) or ' in your data'} split by sentiment: "
              f"{wording.split_words(counts['positive'], counts['neutral'], counts['negative'])}."),
        {"type": "breakdown", "counts": counts},
    ]
    if scope["pending"]:
        blocks.append(_text(f"_{_n(scope['pending'], 'review')} haven't been analysed yet._"))
    if scope["low_confidence"]:
        blocks.append(_text(f"_The model wasn't very sure about {_n(scope['low_confidence'], 'review')}._"))
    blocks.append(_actions("Open in Data view", q))
    return _reply(blocks, _understood(q, "SQL template · overview · no AI used"))


def _trend(q: Query, db_path, today: date) -> dict[str, Any]:
    filters = q.filters(period=False)
    d_from = q.period.date_from if q.period else None
    d_to = q.period.date_to if q.period else None
    if not q.period:
        base = analytics.overview(db_path, **filters)
        if base["total"] == 0:
            return _reply([_text(_nothing_found(q, sentiment=False, db_path=db_path, today=today))], _understood(q, "SQL template · sentiment_trend"))
        d_from, d_to = base["first_day"], base["last_day"]
    span = (date.fromisoformat(d_to) - date.fromisoformat(d_from)).days if d_from and d_to else 0
    gran = q.granularity or analytics.choose_granularity(span)
    note = ""
    if gran == "week" and span <= 7:
        gran, note = "day", "That period is shorter than a full week, so I show one bar per day. "
    rows = analytics.sentiment_trend(gran, db_path, date_from=d_from, date_to=d_to, **filters)
    if not rows:
        return _reply([_text(_nothing_found(q, sentiment=False, db_path=db_path, today=today))], _understood(q, "SQL template · sentiment_trend"))
    rows = analytics.fill_trend(rows, gran, d_from, d_to)
    for r in rows:
        r["label"] = analytics.period_label(r["period"], gran)
    start_word = {"day": "Day", "week": "Week starting (Monday)", "month": "Month"}[gran]
    total = sum(r["total"] for r in rows)
    # "Are refund complaints getting worse?" sets sentiment=negative, but the reading must weigh negative reviews against ALL
    # reviews on that topic: inside an already-negative set "3 of 3 were negative" is always 100% and can never show a change.
    # Without a named period do not pin the dates: d_from/d_to were taken from the NEGATIVE reviews only, which would cut the
    # earlier window short (it would miss a positive review on the topic's first day) and anchor "the last 4 weeks" to the
    # newest negative review instead of the newest review on the topic.
    trend_filters = {**{k: v for k, v in filters.items() if k != "sentiment"},
                     **({"date_from": q.period.date_from, "date_to": q.period.date_to} if q.period else {})}
    label = " ".join(x for x in (q.product, q.topic) if x) + " reviews" if (q.product or q.topic) else "reviews"
    reading = insights.describe_trend(label, insights.recent_vs_earlier(db_path, trend_filters))
    n_all = analytics.overview(db_path, **trend_filters)["total"]
    when = f" {wording.period_phrase(q.period, today)}" if q.period else " across all your data"
    if q.sentiment == "negative" and (q.product or q.topic):
        caption = f"The chart counts only the {total} negative {label} (out of {n_all}), one bar per {gran},{when}:"
    elif q.sentiment:
        caption = f"The chart shows only the {q.sentiment} {label}, one bar per {gran},{when} ({_n(total, 'review')}):"
    else:
        caption = f"Here's how sentiment moved, one bar per {gran},{when} ({_n(total, 'review')} in total):"
    return _reply(
        [
            _text(f"{note}{reading}\n\n{caption}"),
            {"type": "chart", "rows": rows, "x_label": start_word, "gran": gran},
            _actions("Open in Data view", q),
        ],
        _understood(q, f"SQL template · sentiment_trend({gran}) · no AI used"),
    )


def _list(q: Query, db_path, today: date) -> dict[str, Any]:
    limit = q.limit or DEFAULT_LIMIT
    rows, total = database.fetch_reviews(sort=q.order or "newest", page=1, page_size=limit, db_path=db_path, **q.filters())
    if not rows:
        return _reply([_text(_nothing_found(q, db_path=db_path, today=today))], _understood(q, "SQL template · recent_reviews"))
    more = f" ({total:,} in total)" if total > len(rows) else ""
    adjective = {"worst": "lowest-rated", "best": "highest-rated"}.get(q.order or "", "newest")
    how = ("\n\n_I sorted them by star rating first, then by how strongly the model judged each one negative._" if q.order == "worst"
           else "\n\n_I sorted them by star rating first, then by how strongly the model judged each one positive._" if q.order == "best" else "")
    reviews = wording.phrase(q, noun="review" if len(rows) == 1 else "reviews", today=today)
    if q.sentiment:
        reviews = reviews.replace(f"{q.sentiment} ", "", 1)  # "negative" is already in the sentence below
    lead = "Here's the" if len(rows) == 1 else "Here are the"
    noun_sentiment = f"{q.sentiment} " if q.sentiment else ""
    return _reply(
        [
            _text(f"{lead} {len(rows):,} {adjective} {noun_sentiment}{reviews}{more}:{how}" if len(rows) > 1
                  else f"{lead} {adjective} {noun_sentiment}{reviews}{more}:{how}"),
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
            [_text("None of these reviews has a product name, so I can't rank products. "
                   "If you import a CSV with a product column (map it during import), this will work.")],
            _understood(q, "SQL template · by_product"),
        )
    metric = q.sentiment or "total"
    named.sort(key=lambda s: (-s[metric], -s["total"], s["product"]))
    top = named[0]
    kind = f"{metric} reviews" if q.sentiment else "reviews"
    if not top[metric]:
        sentence = f"No product has any {kind} here."
    else:
        tied = [s for s in named if s[metric] == top[metric]]

        def share(s):
            return s[metric] / max(s["positive"] + s["neutral"] + s["negative"], 1)

        leader = max(named, key=lambda s: (share(s), s["total"]))
        if len(tied) > 1:
            sentence = f"**{wording.join_names([s['product'] for s in tied])}** are tied for the most {kind}, with {top[metric]} each."
        else:
            sentence = f"**{top['product']}** has the most {kind} ({top[metric]})"
            if top["total"] == max(s["total"] for s in named) and q.sentiment:
                sentence += (f", but it also has the most reviews overall ({top['total']}), so part of that is just volume "
                             f"({share(top):.0%} of its own reviews are {q.sentiment})")
            sentence += "."
        standout = {"negative": "is hit hardest", "positive": "comes out best"}.get(q.sentiment or "", "stands out most")
        if q.sentiment and (leader["product"] != top["product"] or len(tied) > 1):
            sentence += (f" As a share of its own reviews, **{leader['product']}** {standout}: {leader[metric]} of its "
                         f"{leader['total']} reviews ({share(leader):.0%}) are {q.sentiment}.")
        elif q.sentiment:
            sentence += f" It's also the highest as a share of its own reviews ({top[metric]} of {top['total']}, {share(top):.0%})."
    table = {
        "type": "table",
        "columns": ["Product", "Reviews", "Positive", "Neutral", "Negative", "Negative %"],
        "rows": [[s["product"], s["total"], s["positive"], s["neutral"], s["negative"], f"{s['negative_pct']:.0f}%"] for s in named[:10]],
    }
    blocks = [_text(sentence), table]
    if any(s["total"] < 5 for s in named):
        blocks.append(_text("_A few products have fewer than 5 reviews, so treat their shares as rough._"))
    if without:
        blocks.append(_text(f"_{_n(without, 'review')} without a product name {'is' if without == 1 else 'are'} left out._"))
    blocks.append(_actions("Open in Data view", q))
    return _reply(blocks, _understood(q, f"SQL template · by_product(sorted by {metric}) · no AI used"))


def _average(q: Query, db_path, today: date) -> dict[str, Any]:
    if q.group_by == "product":
        stats = [s for s in analytics.by_product(db_path, **q.filters()) if s["product"] != "(no product)" and s["avg_rating"] is not None]
        if not stats:
            return _reply([_text("None of these products has any star-rated reviews.")], _understood(q, "SQL template · by_product"))
        lowest = q.extreme == "min"
        stats.sort(key=lambda s: ((s["avg_rating"] if lowest else -s["avg_rating"]), -s["rated"], s["product"]))
        reliable = [s for s in stats if s["rated"] >= 3]
        if reliable:
            best = reliable[0]["avg_rating"]
            tied = [s for s in reliable if s["avg_rating"] == best]
            names = wording.join_names([s["product"] for s in tied])
            head = (f"**{names}** {'share' if len(tied) > 1 else 'has'} the {'lowest' if lowest else 'best'} average rating, {best:.2f} out of 5 "
                    "(counting only products with at least 3 star ratings).")
        else:
            head = "No product has 3 or more star ratings yet, so a ranking would be unreliable. Here are the averages anyway:"
        table = {"type": "table", "columns": ["Product", "Average rating", "Rated reviews", "All reviews", "Negative %"],
                 "rows": [[s["product"], f"{s['avg_rating']:.2f} / 5", s["rated"], s["total"], f"{s['negative_pct']:.0f}%"] for s in stats[:10]]}
        blocks = [_text(head), table]
        if any(s["rated"] < s["total"] for s in stats):
            blocks.append(_text("_Only reviews with a star rating count towards the average, so it can differ from the sentiment of all "
                                "reviews (see the Negative % column)._"))
        blocks.append(_actions("Open in Data view", q))
        return _reply(blocks, _understood(q, "SQL template · by_product · no AI used"))
    o = analytics.overview(db_path, **q.filters())
    if o["total"] == 0:
        return _reply([_text(_nothing_found(q, db_path=db_path, today=today))], _understood(q, "SQL template · overview"))
    if o["avg_rating"] is None:
        return _reply([_text(f"None of the {_n(o['total'], 'matching review')} has a star rating.")], _understood(q, "SQL template · overview"))
    return _reply(
        [_text(f"The average rating is **{o['avg_rating']:.2f} out of 5**, across the reviews that have a star rating"
               f"{wording.rest(q, sentiment=True, today=today)}."), _actions("Open in Data view", q)],
        _understood(q, "SQL template · overview(avg_rating) · no AI used"),
    )


def _compare_named_products(q: Query, names: list[str], db_path, today: date) -> dict[str, Any]:
    """"How is Blender Pro doing compared to Desk Lamp?": the named products side by side."""
    base = {k: v for k, v in q.filters(sentiment=False, exclude=False).items() if k not in ("product", "search", "topic")}
    when = wording.rest(Query(period=q.period), sentiment=False, today=today).strip() if q.period else ""
    stats = []
    for name in names:
        f = {**base, "product": name}
        o = analytics.overview(db_path, **f)
        complaint = next((t["topic"] for t in analytics.topic_breakdown(db_path, **{**f, "sentiment": "negative"}) if t["topic"] != "other"), None)
        stats.append({"name": name, "o": o, "complaint": complaint})
    if all(s["o"]["total"] == 0 for s in stats):
        return _reply([_text(f"I couldn't find any reviews for {wording.join_names(names)}{' ' + when if when else ''}. "
                             + _coverage(db_path).strip())], f"SQL template · overview x{len(names)} · compare products")
    stats.sort(key=lambda s: (-s["o"]["positive_pct"], s["o"]["negative_pct"]))
    top, bottom = stats[0], stats[-1]
    po, bo = top["o"], bottom["o"]
    if po["total"] == 0 or bo["total"] == 0:
        empty = [s["name"] for s in stats if s["o"]["total"] == 0]
        head = f"There are no reviews for {wording.join_names(empty)}{' ' + when if when else ''}, so I can't compare them."
    elif po["positive_pct"] - bo["positive_pct"] < 5:
        head = (f"**{wording.join_names([s['name'] for s in stats])}** are doing about the same: "
                + "; ".join(f"{s['name']} {s['o']['positive_pct']:.0f}% positive and {s['o']['negative_pct']:.0f}% negative" for s in stats)
                + f"{' ' + when if when else ''}.")
    else:
        head = (f"**{top['name']}** is doing better than **{bottom['name']}**{' ' + when if when else ''}. "
                f"{po['positive_pct']:.0f}% of its reviews are positive against {bo['positive_pct']:.0f}%, "
                f"and {po['negative_pct']:.0f}% are negative against {bo['negative_pct']:.0f}%.")
        if po["avg_rating"] is not None and bo["avg_rating"] is not None and po["avg_rating"] != bo["avg_rating"]:
            head += (f" Its average rating is {'higher' if po['avg_rating'] > bo['avg_rating'] else 'lower'} too "
                     f"({po['avg_rating']:.2f} against {bo['avg_rating']:.2f} out of 5).")
    table = {"type": "table", "columns": ["Product", "Reviews", "Positive %", "Negative %", "Avg rating", "Most common complaint"],
             "rows": [[s["name"], s["o"]["total"], f"{s['o']['positive_pct']:.0f}%", f"{s['o']['negative_pct']:.0f}%",
                       f"{s['o']['avg_rating']:.2f}" if s["o"]["avg_rating"] is not None else "n/a", s["complaint"] or "none"] for s in stats]}
    blocks = [_text(head), table]
    if any(0 < s["o"]["total"] < 5 for s in stats):
        blocks.append(_text("_One of the products has fewer than 5 reviews, so treat this as weak evidence._"))
    blocks.append(_actions("Open in Data view", Query(period=q.period)))
    return _reply(blocks, f"SQL template · overview x{len(names)} · no AI used · understood as: compare {wording.join_names(names)}"
                          f"{' ' + q.period.label if q.period else ' overall'}")


def _owner_question(spec: dict[str, Any], question: str, db_path, today: date, summarizer, analyzer) -> dict[str, Any]:
    """Decision questions ("which product should I promote?", "price complaints vs delivery complaints?"); see owner_questions.py."""
    kind, period = spec["kind"], spec.get("period")
    if kind == "query":  # an ordinary list / "why" question that only needed rewording
        q: Query = spec["query"]
        result = _answer(q, question, db_path, today, summarizer, analyzer)
        if q.intent in chat_parser.READ_INTENTS:
            result["query"] = q
        return result
    note = f"SQL template · {kind} · no AI used"
    if kind in ("promote", "improve"):
        return _reply(owner_questions.recommend(kind, db_path, period, today), note)
    if kind == "mixed":
        return _reply(owner_questions.mixed(db_path, period, today), note)
    a, b = spec["topics"]
    if kind == "topics":
        return _reply(owner_questions.compare_topics(a, b, spec["negative"], db_path, period, today), note)
    blocks, tagged = owner_questions.both_topics(a, b, spec["negative"], db_path, period, today)
    if tagged:
        blocks += [_text("Here are the newest ones that mention both:"), {"type": "reviews", "rows": [display_row(r) for r in tagged]}]
    return _reply(blocks, note)


def _bare(label: str) -> str:
    return label[3:] if label.startswith("in ") else label


def _compare_any(q: Query, db_path, today: date) -> dict[str, Any]:
    return _compare_products(q, db_path, today) if q.group_by == "product" else _compare(q, db_path, today)


def _direction_sentence(result: dict[str, Any], direction: str | None) -> str:
    """"Which product improved the most?": the product whose share of negative reviews moved the most in that direction."""
    if direction not in ("better", "worse"):
        return ""
    rows = [r for r in result["rows"] if r["product"] != product_change.NO_PRODUCT and not r["few"]]
    if not rows:
        return "_No product has enough reviews in both halves to say which one changed most._"
    for r in rows:
        r["shift"] = r["negative_pct"][1] - r["negative_pct"][0]
    rows.sort(key=lambda r: (r["shift"] if direction == "better" else -r["shift"], r["product"]))
    top = rows[0]
    moved = top["shift"] < 0 if direction == "better" else top["shift"] > 0
    if not moved or abs(top["shift"]) < 3:
        return f"**No product clearly {'improved' if direction == 'better' else 'got worse'}**: the share of negative reviews barely moved for any product with enough reviews."
    verb = "improved the most" if direction == "better" else "got worse the most"
    return (f"**{top['product']}** {verb}: its share of negative reviews went from {top['negative_pct'][0]:.0f}% to {top['negative_pct'][1]:.0f}% "
            f"({top['negative'][0]} to {top['negative'][1]} negative reviews).")


def _split_open_period(q: Query, db_path) -> None:
    """"Which product improved the most since July?": there is no earlier period to compare with, so split the reviews from
    July on into an earlier and a later half by date and compare those. (Without this the "earlier" window fell before the data.)"""
    if q.previous_period or q.period is None or not q.period.label.lower().startswith("since"):
        return
    first, last = database.get_date_range(db_path)
    if not first or not last:
        return
    start = max(date.fromisoformat(q.period.date_from), date.fromisoformat(first[:10]))
    end = min(date.fromisoformat(q.period.date_to), date.fromisoformat(last[:10]))
    if (end - start).days < 13:
        return
    mid = start + (end - start) // 2
    q.previous_period = chat_parser.Period(start.isoformat(), mid.isoformat(), f"{_fmt_day(start)} – {_fmt_day(mid)}")
    q.period = chat_parser.Period((mid + timedelta(days=1)).isoformat(), end.isoformat(), f"{_fmt_day(mid + timedelta(days=1))} – {_fmt_day(end)}")


def _compare_products(q: Query, db_path, today: date) -> dict[str, Any]:
    """"Which product caused the increase in positive reviews in October?": per-product change between two periods.
    Reviews show WHAT moved for each product, never WHY, so the wording says "accounts for", not "caused"."""
    _split_open_period(q, db_path)
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
        return _reply([_text(f"There aren't any reviews in either period ({_bare(cur.label)} and {_bare(prev_label)})." + _coverage(db_path))], note)
    if before == 0 or after == 0:
        empty = prev_label if before == 0 else cur.label
        return _reply([_text(f"I can't tell what changed for each product, because there aren't any reviews in {_bare(empty)}." + _coverage(db_path))], note)
    focus = q.sentiment if q.sentiment in ("positive", "negative") else None
    lines = product_change.describe(result, focus)
    verdict = _direction_sentence(result, q.direction)
    if verdict:
        lines.insert(0, verdict)
    blocks: list[dict] = [
        _text(f"Comparing **{_bare(cur.label)}** with **{_bare(prev_label)}**:\n\n" + "\n\n".join(lines)),
        {"type": "table", "columns": product_change.TABLE_COLUMNS, "rows": product_change.table_rows(result)},
        _text("_These numbers show what changed for each product, not why. Counts also move when more or fewer reviews come in, "
              "so look at the negative-share column too._"),
    ]
    if any(r["few"] for r in result["rows"]):
        blocks.append(_text("_Products marked “few reviews” have fewer than 5 in one of the periods, so I don't name them as drivers._"))
    blocks.append(_actions("Open in Data view", q))
    return _reply(blocks, note)


def _range_name(date_from: str, date_to: str) -> str:
    """"September" for a whole calendar month, otherwise "Oct 1 – Oct 2"."""
    start, end = date.fromisoformat(date_from), date.fromisoformat(date_to)
    if start.day == 1 and end.month == start.month and end.year == start.year and (end + timedelta(days=1)).day == 1:
        return f"{start:%B}" + (f" {start.year}" if start.year != date.today().year else "")
    return f"{_fmt_day(start)} – {_fmt_day(end)}"


def _compare(q: Query, db_path, today: date) -> dict[str, Any]:
    filters = q.filters(period=False, sentiment=False)
    _split_open_period(q, db_path)
    cur = q.period or chat_parser.parse_periods("last 7 days", today)[0]
    if q.previous_period:
        current = analytics.overview(db_path, date_from=cur.date_from, date_to=cur.date_to, **filters)
        previous = analytics.overview(db_path, date_from=q.previous_period.date_from, date_to=q.previous_period.date_to, **filters)
        prev_label = q.previous_period.label
    else:
        # same rule as the per-product comparison: a whole month is compared with the month before it
        p_from, p_to = product_change.previous_window(cur.date_from, cur.date_to)
        current = analytics.overview(db_path, date_from=cur.date_from, date_to=cur.date_to, **filters)
        previous = analytics.overview(db_path, date_from=p_from, date_to=p_to, **filters)
        prev_label = _range_name(p_from, p_to)
    cur_name = _range_name(cur.date_from, cur.date_to)
    prev_name = _range_name(q.previous_period.date_from, q.previous_period.date_to) if q.previous_period else prev_label
    if current["total"] == 0 and previous["total"] == 0:
        return _reply([_text("There aren't any reviews in either period." + _coverage(db_path))], _understood(q, "SQL template · compare"))
    if current["total"] == 0 or previous["total"] == 0:
        empty = cur.label if current["total"] == 0 else prev_label
        text = (f"I can't tell whether anything changed, because there aren't any reviews {empty if empty.startswith('in ') else 'in ' + empty}."
                + _coverage(db_path))
    else:
        metric = q.sentiment if q.sentiment in ("positive", "negative") else "negative"
        n_now, n_before = current[metric], previous[metric]
        share_now, share_before = current[f"{metric}_pct"], previous[f"{metric}_pct"]
        verb = "went up" if n_now > n_before else "went down" if n_now < n_before else "stayed the same"
        text = (f"{metric.capitalize()} reviews **{verb}**, from {n_before} in {prev_name} ({share_before:.0f}% of {_n(previous['total'], 'review')}) "
                f"to {n_now} in {cur_name} ({share_now:.0f}% of {_n(current['total'], 'review')}).")
        count_dir, share_dir = (n_now > n_before) - (n_now < n_before), (round(share_now) > round(share_before)) - (round(share_now) < round(share_before))
        if count_dir and share_dir and count_dir != share_dir:
            text += (f" The count and the share moved in opposite directions because the number of reviews changed too "
                     f"({previous['total']} → {current['total']}), so the share is the fairer comparison.")
        elif count_dir and not share_dir:
            text += " As a share of all reviews it hardly moved, so most of the change is just more or fewer reviews coming in."
        if min(current["total"], previous["total"]) < 5:
            text = "Only a rough read, because one period has very few reviews: " + text[0].lower() + text[1:]
    table = {
        "type": "table",
        "columns": ["Period", "Reviews", "Positive %", "Negative %", "Avg rating"],
        "rows": [
            [cur_name, current["total"], f"{current['positive_pct']:.0f}%", f"{current['negative_pct']:.0f}%", current["avg_rating"] if current["avg_rating"] is not None else "n/a"],
            [prev_name, previous["total"], f"{previous['positive_pct']:.0f}%", f"{previous['negative_pct']:.0f}%", previous["avg_rating"] if previous["avg_rating"] is not None else "n/a"],
        ],
    }
    blocks: list[dict] = [_text(text), table]
    if min(current["total"], previous["total"]) < 5:
        blocks.append(_text("_One of the periods has fewer than 5 reviews, so treat this as weak evidence._"))
    blocks.append(_actions("Open in Data view", q))
    return _reply(blocks, _understood(q, "SQL template · overview x2 · no AI used"))


def _inside_peak_month(q: Query, db_path, month: str, metric: str, pick, name: str = "that month") -> str:
    """"When was the peak?" deserves more than a month: name the busiest week and day inside the peak month."""
    first = date.fromisoformat(month + "-01")
    last = (first.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
    filters = {**q.filters(period=False, sentiment=False), "date_from": first.isoformat(), "date_to": last.isoformat()}
    days = analytics.sentiment_trend("day", db_path, **filters)
    weeks = analytics.sentiment_trend("week", db_path, **filters)
    if len(days) < 3 or not weeks:
        return ""
    kind = f"{metric} review" if metric != "total" else "review"
    out = ""
    best_week = pick(weeks, key=lambda r: r[metric])
    if best_week[metric] > 0:
        monday = date.fromisoformat(analytics.period_label(best_week["period"], "week"))
        out += f" Within {name}, the week starting {_fmt_day(monday)} had the most ({best_week[metric]})"
    best_day = pick(days, key=lambda r: r[metric])
    tied = [d for d in days if d[metric] == best_day[metric]]
    if best_day[metric] > 1 and len(tied) == 1:
        out += (", and " if out else f" Within {name}, ") + f"the single busiest day was {_fmt_day(date.fromisoformat(best_day['period']))} ({best_day[metric]} {kind}s)"
    return out + "." if out else ""


def _peak(q: Query, db_path, today: date) -> dict[str, Any]:
    gran = q.granularity or "month"
    base = analytics.overview(db_path, **q.filters(sentiment=False))
    if base["total"] == 0:
        return _reply([_text(_nothing_found(q, sentiment=False, db_path=db_path, today=today))], _understood(q, "SQL template · sentiment_trend"))
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
        if gran == "day":
            try:
                d = date.fromisoformat(label)
                return f"{d:%A}, {d:%b} {d.day}, {d.year}"
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
    if gran == "month" and len(winners) == 1 and re.fullmatch(r"\d{4}-\d{2}", str(winners[0]["period"])):
        text += _inside_peak_month(q, db_path, str(winners[0]["period"]), metric, pick, nice(winners[0]).split()[0])
    if q.sentiment and len(winners) == 1 and rows:
        w = winners[0]
        if w["total"] == max(r["total"] for r in rows) and len(rows) > 1:
            text += f" It was also your busiest {gran} overall ({w['total']} reviews), so part of that is just volume."
        eligible = [r for r in rows if r["total"] >= 10]
        if eligible:
            best = max(eligible, key=lambda r: (r[metric] / r["total"], r["total"]))
            if best is not w:
                rank_word = {"negative": "the worst", "positive": "the best"}.get(metric, "the highest")
                text += (f" As a share of each {gran}'s own reviews, **{nice(best)}** was {rank_word}: {best[metric]} of {best['total']} "
                         f"({best[metric] / best['total']:.0%})" + (", although your data only covers part of that month" if partial(best) else "") + ".")
    partials = [nice(r) for r in rows if partial(r)]
    table = {"type": "table", "columns": [gran.capitalize(), "Reviews", "Positive", "Neutral", "Negative", "Negative %"],
             "rows": [[nice(r) + (" (partial)" if partial(r) else ""), r["total"], r["positive"], r["neutral"], r["negative"],
                       f"{r['negative'] / max(r['positive'] + r['neutral'] + r['negative'], 1):.0%}"] for r in rows[-12:]]}
    blocks = [_text(text), table]
    if partials:
        blocks.append(_text(f"_Your data only covers part of {' and '.join(partials)}, so {'its' if len(partials) == 1 else 'their'} totals are lower than a full month's._"))
    if len(rows) < 3:
        blocks.append(_text(f"_Your data only covers {_n(len(rows), gran)}, so this comparison isn't very meaningful._"))
    blocks.append(_actions("Open in Data view", q))
    return _reply(blocks, _understood(q, f"SQL template · sentiment_trend({gran}) · no AI used"))


def _mismatch(q: Query, db_path, today: date) -> dict[str, Any]:
    stats = [s for s in analytics.by_product(db_path, **q.filters(sentiment=False))
             if s["product"] != "(no product)" and s["avg_rating"] is not None]
    rows, _ = database.fetch_reviews(sort="newest", page=1, page_size=500, db_path=db_path, **q.filters(sentiment=False))
    rated = [r for r in rows if r.get("rating") is not None and r.get("sentiment_label")]
    if not rated:
        return _reply([_text("None of these reviews has both a star rating and a sentiment, so I can't compare them.")],
                      _understood(q, "SQL template · rating vs sentiment"))
    clash = [r for r in rated if (r["sentiment_label"] == "positive" and r["rating"] <= 2)
             or (r["sentiment_label"] == "negative" and r["rating"] >= 4)]
    flagged = [s["product"] for s in stats if s["rated"] >= 2 and s["positive_pct"] >= 50 and s["avg_rating"] <= 3]
    head = (f"Of the {_n(len(rated), 'review')} that have both a star rating and a sentiment, **{len(clash)}** clearly disagree: "
            "positive wording with 1–2 stars, or negative wording with 4–5 stars.")
    if flagged:
        head += f" These products are mostly positive but average 3 stars or less: **{wording.join_names(flagged)}**."
    else:
        head += " No product is mostly positive while averaging 3 stars or less."
        close = [s for s in stats if s["rated"] >= 5 and s["positive_pct"] >= 40]
        if close:
            c = min(close, key=lambda s: s["avg_rating"])
            head += f" The closest is **{c['product']}**, with {c['positive_pct']:.0f}% positive reviews and a {c['avg_rating']:.1f}-star average."
    table = {"type": "table", "columns": ["Product", "Positive %", "Average rating", "Rated reviews"],
             "rows": [[s["product"], f"{s['positive_pct']:.0f}%", f"{s['avg_rating']:.2f}", s["rated"]] for s in stats[:10]]}
    blocks = [_text(head), table]
    if clash:
        blocks += [_text("Here are the newest reviews where the stars and the wording disagree:"),
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


def _opinion_answer(verdict: str, asks_unhappy: bool) -> str:
    """A short Yes / No / Mixed to open "are customers happy with X?" before the numbers."""
    happy = {"mostly positive": "Yes, mostly.", "leaning positive": "Mostly yes, though not everyone is happy.",
             "mixed": "It's mixed.", "leaning negative": "Not really: more reviews are negative than positive.",
             "mostly negative": "No, most of the reviews are negative.", "mostly neutral": "Not strongly either way: most reviews are neutral."}
    unhappy = {"mostly positive": "No, most of the reviews are positive.", "leaning positive": "Not really: more reviews are positive than negative.",
               "mixed": "It's mixed.", "leaning negative": "Somewhat, more reviews are negative than positive.",
               "mostly negative": "Yes, most of the reviews are negative.", "mostly neutral": "Not strongly either way: most reviews are neutral."}
    return (unhappy if asks_unhappy else happy).get(verdict, "It's mixed.")


def _issue_sentence(q: Query, focus: str, ranked: list[dict], base_filters: dict, db_path) -> str:
    """"The problems people mention most are rude staff (6 of 10), ...": the specific complaints inside a topic."""
    topics_to_read = [q.topic] if q.topic else [t["topic"] for t in ranked[:2]]
    topics_to_read = [t for t in topics_to_read if t in issues_service.ISSUES]
    if not topics_to_read:
        return ""
    rows, _ = database.fetch_reviews(sort="newest", page=1, page_size=500, db_path=db_path, sentiment=focus, **base_filters)
    parts: list[str] = []
    for topic in topics_to_read:
        mine = [r for r in rows if topic in topics_service.from_storage(r.get("topics"))]
        found, n = issues_service.common_issues(topic, [r.get("review_text") or r.get("text") or "" for r in mine])
        if not found:
            continue
        listed = ", ".join(f"{label} ({c} of {n})" if i == 0 else f"{label} ({c})" for i, (label, c) in enumerate(found))
        listed = listed.replace(", ", ", ", 1)
        parts.append(f"The problems people mention most are {listed}." if q.topic else f"On {topic}, the problems people mention most are {listed}.")
    return " ".join(parts)


def _reasons(q: Query, db_path, summarizer: Summarizer | None, recommend: bool, question: str = "") -> dict[str, Any]:
    opinion = chat_parser.is_opinion(question)
    focus = "negative" if opinion else (q.sentiment or "negative")  # "are customers happy with X": the issues are the unhappy side
    # "Leaving out delivery" here means: do not rank delivery as an issue. Reviews that mention delivery AND something
    # else still count for that something else, so the reviews themselves are not filtered away.
    base_filters = q.filters(sentiment=False, exclude=False)
    scope = analytics.overview(db_path, **base_filters)
    scope_text = q.scope(sentiment=False, exclusions=False).replace("reviews", "", 1).strip() or "overall"
    tail = wording.rest(q, sentiment=False, exclusions=False).strip()  # "about delivery in October"
    if scope["total"] == 0:
        return _reply(
            [_text(f"I can't answer that from your data, because there aren't any reviews {tail or 'that match'}."
                   + (" I searched the review text and product names for those words." if q.search_terms else ""))],
            _understood(q, "Retrieval found nothing · abstained"),
        )
    n_focus = scope[focus]
    if n_focus == 0:
        newest, _ = database.fetch_reviews(sort="newest", page=1, page_size=3, db_path=db_path, **base_filters)
        text = (f"None of the {_n(scope['total'], 'matching review')}{' ' + tail if tail else ''} {'is' if scope['total'] == 1 else 'are'} {focus}: "
                f"{wording.split_words(scope['positive'], scope['neutral'], scope['negative'])}.")
        blocks0: list[dict] = [_text(text)]
        if newest:
            blocks0 += [_text("Here are the newest matching reviews:"), {"type": "reviews", "rows": [display_row(r) for r in newest]}]
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
    where = tail or "in your data"
    analysed = max(scope["positive"] + scope["neutral"] + scope["negative"], 1)
    if scope["total"] >= 5 and n_focus >= 2:
        facts_text = (f"{insights.one_in(n_focus, analysed).capitalize()} of the {_n(scope['total'], 'review')} {where} are {focus} "
                      f"({n_focus}, or {n_focus / analysed:.0%}).")
    else:
        facts_text = f"Of the {_n(scope['total'], 'review')} {where}, {n_focus} {'is' if n_focus == 1 else 'are'} {focus} ({n_focus / analysed:.0%})."
    if opinion:  # lead with the answer to "do customers think X is fair / are they happy with X"
        pos, neu = scope["positive"], scope["neutral"]
        # "mostly" needs a clear lead: half positive but as many negative is mixed, not "mostly positive"
        verdict = ("mostly positive" if pos / analysed >= 0.5 and pos >= 2 * n_focus else "mostly negative" if n_focus / analysed >= 0.5 and n_focus >= 2 * pos
                   else "mostly neutral" if neu / analysed >= 0.5
                   else "leaning positive" if pos / analysed >= 0.4 and pos > n_focus else "leaning negative" if n_focus / analysed >= 0.4 and n_focus > pos
                   else "mixed")
        subject = q.topic or q.product or (q.search_terms[0] if q.search_terms else None)
        asks_unhappy = bool(re.search(r"\b(unhappy|dissatisfied|disappointed|upset|annoyed)\b", question.lower()))
        answer = _opinion_answer(verdict, asks_unhappy)
        facts_text = (f"**{answer}** Of the {_n(scope['total'], 'review')} {where}, "
                      f"{pos} {'is' if pos == 1 else 'are'} positive ({pos / analysed:.0%}), {neu} neutral ({neu / analysed:.0%}) "
                      f"and {n_focus} negative ({n_focus / analysed:.0%}).")
    issue_text = _issue_sentence(q, focus, ranked, base_filters, db_path) if focus == "negative" else ""
    if issue_text:
        facts_text += " " + issue_text
    if ranked:
        items = [f"{t['topic']} ({t[focus]} of {n_focus})" if i == 0 else f"{t['topic']} ({t[focus]})" for i, t in enumerate(ranked[:3])]
        listed = items[0] if len(items) == 1 else ", ".join(items[:-1]) + f" and {items[-1]}"
        lead = ("Among the negative ones, the other things people mention are" if opinion and q.topic
                else "Among the negative ones, the issues that come up most are" if opinion
                else f"Some of them also mention" if q.topic and issue_text
                else f"Among those {n_focus} {focus} reviews, the other things people mention are" if q.topic
                else "The topics that come up most in the negative reviews are" if focus == "negative" and issue_text
                else "The issues that come up most in the negative reviews are" if focus == "negative"
                else f"The things customers mention most in the {focus} reviews are")
        facts_text += f" {lead} {listed}; a review can mention more than one thing."
    if q.exclude_topics:
        shown = ", ".join(f"{t} ({left_out.get(t, 0)} of {n_focus})" for t in q.exclude_topics)
        facts_text += f" I've left out {shown}, as you asked."
        if not ranked:
            facts_text += " Nothing else stands out in these reviews."
    if (q.topic or q.product or q.search_terms) and not opinion:  # an opinion answer already states the split
        facts_text += f" For reference: {scope['positive']} positive, {scope['neutral']} neutral and {scope['negative']} negative."
    evidence = analytics.examples(4, db_path, sentiment=focus, **base_filters)
    evidence_rows = [display_row(r) for r in evidence]

    blocks: list[dict] = []
    if scope["total"] < 5:
        blocks.append(_text(f"_Only {_n(scope['total'], 'review')} match{'es' if scope['total'] == 1 else ''}, so this is weak evidence._"))

    body = None
    evidence_head, evidence_label = f"{_n(n_focus, focus + ' review')}", f"showing the {len(evidence_rows)} newest"
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
                           "text": "This summary was written by AI from a sample of the matching reviews. The numbers are calculated from your data; check the cited reviews under Evidence."})
            body = f"{facts_text}\n\n{body}"
            route_note = "Retrieval → language model → checked against the data · answer shows its evidence"
            cited = set(facts.get("cited_ids") or [])
            if cited:
                evidence_rows = [r for r in sample if r["id"] in cited] or evidence_rows
                evidence_head, evidence_label = f"{_n(scope['total'], 'matching review')}", f"showing the {len(evidence_rows)} the summary cites"
        else:
            blocks.append({"type": "badge", "kind": "rule",
                           "text": "The AI summary wasn't available or couldn't be checked against your data, so you're seeing the calculated summary instead."
                                   + (f" (Reason: {str(facts['problem'])[:200]})" if facts.get("problem") else "")})
            route_note = "Rule-based fallback · language model unavailable or its answer failed a check"
    elif summarizer is None and chat_parser.wants_summary(question):
        blocks.append({"type": "badge", "kind": "rule",
                       "text": "AI summaries are switched off (no Gemini key or model, or AI_SUMMARIES=0), so you're seeing the calculated summary instead of a written one."})
        route_note = "Rule-based summary · AI summaries are off"
    if not body:
        body = facts_text
        if q.topic and not ranked and focus == "negative" and re.match(r"\s*why\b", question.lower()):
            body += (f"\n\nA heads-up: the topic tags only tell me these reviews are about **{q.topic}**, not what went wrong. The specific reasons "
                     "are in the reviews below, in the customers' own words. If you'd like a written explanation, ask me to **summarise** them.")
        details = insights.topic_details(db_path, base_filters, focus, [t["topic"] for t in ranked[:3]]) if (ranked and recommend) else {}
        if top and len(tied) > 1 and not opinion and not q.topic:
            names = ", ".join(t["topic"] for t in tied[:-1]) + f" and {tied[-1]['topic']}"
            body += (f"\n\n**{names[:1].upper() + names[1:]}** are neck and neck at the top ({top[focus]} each), so there's no single biggest reason."
                     if focus == "negative" else
                     f"\n\n**{names[:1].upper() + names[1:]}** are tied at the top ({top[focus]} each), so there's no single standout.")
        elif top and not recommend and not opinion and not q.topic:  # not for "are customers happy with X": there it answers a different question
            body += (f"\n\nThe biggest single issue is **{top['topic']}** ({top[focus]} of the {n_focus} {focus} reviews)."
                     if focus == "negative" else
                     f"\n\nThe thing customers {'like most' if focus == 'positive' else 'mention most'} is **{top['topic']}** "
                     f"({top[focus]} of the {n_focus} {focus} reviews).")
        if top and recommend:
            candidates = tied if len(tied) > 1 else [top]
            order = sorted(candidates, key=lambda t: (not details.get(t["topic"], {}).get("rising", False), -t[focus], t["topic"]))
            pick = order[0]
            d = details.get(pick["topic"], {})
            text = f"**My suggestion:** start with **{pick['topic']}**, which comes up in {pick[focus]} of the {n_focus} {focus} reviews"
            if len(candidates) > 1 and d.get("rising"):
                text += f" (tied with {wording.join_names([t['topic'] for t in candidates if t is not pick])}, but it's the one getting worse)"
            text += "."
            if d.get("top_product") and d["top_product"][1] >= 2:
                text += f" Most of those are about **{d['top_product'][0]}** ({d['top_product'][1]})."
            if d.get("rising"):
                text += f" {d['recent']} of them came in the last {d['window']}, more than you'd expect, so it's getting worse."
            others = [t for t in ranked[:3] if t is not pick]
            if others:
                o = others[0]
                od = details.get(o["topic"], {})
                text += f" After that, look at **{o['topic']}** ({o[focus]})" + (f", mostly about {od['top_product'][0]}" if od.get("top_product") and od["top_product"][1] >= 2 else "") + "."
            body += "\n\n" + text
    blocks.append(_text(body))
    if evidence_rows:
        blocks.append({
            "type": "evidence",
            "scope": f"{evidence_head}{' ' + tail if tail else ''} · {evidence_label}",
            "rows": evidence_rows,
        })
    blocks.append(_actions("Open in Data view", q))
    return _reply(blocks, _understood(q, route_note))
