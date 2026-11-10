"""Questions a business owner asks to DECIDE something, which the general question types did not cover.

  - "Which product is safest to promote?" / "Which product should I stop selling or improve first?"
  - "Are our price complaints bigger than our delivery complaints?"   (one topic against another)
  - "Are customers who complain about delivery also unhappy with the product itself?"   (reviews that mention both)
  - "Which products have mixed reviews?"
  - "Give me 3 examples of customers complaining about packaging."   (becomes an ordinary review list)
  - "Which issue is hurting Smart Speaker the most?"   (becomes an ordinary "why" about that product)

They are recognised by fixed rules (no AI call, same answer every time). Every number is counted from the reviews with SQL; the
wording only says what the numbers show. Reviews cannot show profit or stock, so recommendations say they rest on opinions only.
"""
from __future__ import annotations

import math
import re
from datetime import date
from typing import Any

import database
from services import analytics, chat_parser, issues as issues_service, topics as topics_service, wording
from services.chat_parser import Query

_PRODUCT_WORDS = r"(?:products?|items?)"
_IMPROVE = re.compile(
    r"stop selling|discontinu\w*|delist\w*|\bdrop(?:ping)? (?:a|the|which|any)\b|(?:remove|pull|retire|cut)\w* (?:a |the |any )?(?:product|item)"
    r"|improve (?:first|most)|fix (?:first|most)|needs? (?:the )?(?:most )?(?:improv\w*|fix\w*|attention|work)|worst (?:product|item)"
    r"|weakest (?:product|item)|underperform\w*|problem(?:atic)? (?:product|item)|which (?:product|item)s? (?:is|are) (?:hurting|failing|letting)")
_PROMOTE = re.compile(r"promot\w*|\bpush\b|advertis\w*|\bmarket(?:ing)?\b|highlight|showcase|campaign|best[- ]?sell\w*|safest (?:bet|choice|product)|hero product|flagship")
_MIXED = re.compile(r"\bmixed (?:reviews?|feedback|opinions?|reactions?|ratings?)\b|\b(?:divisive|polari[sz]ing)\b|\bsplit (?:opinions?|reviews?)\b")
_EXAMPLES = re.compile(r"\b(?:examples?|samples?|instances?)\b")
_ISSUE_HURTING = re.compile(r"\bwhich (?:issues?|problems?|complaints?)\b.{0,25}\b(?:hurting|affecting|hitting|dragging|causing)\b")
_COMPARE_CUE = re.compile(r"\b(?:bigger|larger|more|worse|greater|higher|lower|fewer|less|compare[sd]?|comparison|versus|vs\.?|than|outweigh\w*|or)\b")
_BOTH_CUE = re.compile(r"\balso\b|\bboth\b|\bas well\b|\btoo\b|\bsame (?:customers?|people|reviews?)\b|\band (?:the )?product itself\b")
_NUMBER = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
_MIN_REVIEWS = 8  # a product needs this many reviews before it can be recommended or flagged


def topics_in(ql: str) -> list[str]:
    """Topics named in the question, in the order they appear ("price ... delivery" -> ["price", "delivery"])."""
    found: list[tuple[int, str]] = []
    for topic, pattern in chat_parser.TOPIC_PATTERNS.items():
        m = re.search(pattern, ql)
        if m:
            found.append((m.start(), topic))
    m = re.search(r"\b(?:product|item) itself\b|\bthe (?:product|item)s? (?:themselves|quality)\b", ql)
    if m and not any(t == "product quality" for _, t in found):
        found.append((m.start(), "product quality"))
    return [t for _, t in sorted(found)]


def _period(question: str, today: date):
    periods = chat_parser.parse_periods(question.lower(), today)
    return periods[0] if len(periods) == 1 else None


def detect(question: str, products: list[str], today: date) -> dict[str, Any] | None:
    """Which of these question types it is, or None. Returns {"kind": ..., ...} for the router."""
    ql = re.sub(r"\s+", " ", (question or "").lower()).strip()
    if not ql or len(ql) > 400:
        return None
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", ql) if s.strip()]
    ask = sentences[-1] if len(sentences) > 1 else ql  # "I'm launching a sale next week. Which product is safest?": the question is the last sentence
    period = _period(ask, today)
    named = chat_parser.named_products(question, products)

    if re.search(r"\b(?:which|what)\b", ask) and re.search(_PRODUCT_WORDS, ask):
        if _IMPROVE.search(ask):
            return {"kind": "improve", "period": period}
        if _PROMOTE.search(ask):
            return {"kind": "promote", "period": period}
    if _IMPROVE.search(ask) and re.search(r"\bwhich\b", ask):
        return {"kind": "improve", "period": period}
    if re.search(r"\b(?:which|what)\b", ask) and _MIXED.search(ask) and not named:
        return {"kind": "mixed", "period": period}

    if _EXAMPLES.search(ask) and re.search(r"\b(?:give|show|provide|list|pull|find|share|need|want|get)\b", ask):
        n = next((int(w) for w in re.findall(r"\b\d+\b", ask)[:1]), None) or next((v for k, v in _NUMBER.items() if re.search(rf"\b{k}\b", ask)), None)
        n = max(1, min(n or 3, 10))
        rewritten = _EXAMPLES.sub("", re.sub(r"\b(?:give me|show me|provide|list|pull|find|share|i need|i want|can you give me|can you show me|please)\b", "", ask))
        rewritten = re.sub(r"\b(?:of|from|for)\s+customers?\b", "of", rewritten)
        subject = re.sub(r"\b(?:customers?|people|buyers)\b", "", rewritten).strip(" .?")
        q = chat_parser.parse(f"show me {n} {subject} reviews", products, today)
        q.intent, q.limit, q.confident = "list", n, True
        if q.sentiment is None and re.search(chat_parser._NEG, ask):
            q.sentiment = "negative"
        return {"kind": "query", "query": q}

    if named and len(named) == 1 and _ISSUE_HURTING.search(ask):
        q = Query(intent="reasons", sentiment="negative", product=named[0], period=period, confident=True)
        return {"kind": "query", "query": q}

    tps = topics_in(ask)
    if len(tps) >= 2 and not named:
        if _BOTH_CUE.search(ask) and re.search(r"\b(?:who|that|which)\b", ask):
            return {"kind": "both", "topics": tps[:2], "period": period, "negative": bool(re.search(chat_parser._NEG, ask)) or "unhappy" in ask}
        if _COMPARE_CUE.search(ask):
            return {"kind": "topics", "topics": tps[:2], "period": period,
                    "negative": bool(re.search(chat_parser._NEG, ask)) or bool(re.search(r"complain", ask))}
    return None


# --------------------------------------------------------------------------- numbers
def _wilson_low(k: int, n: int, z: float = 1.64) -> float:
    """Lower end of the likely range for a share k/n: a small sample cannot look better than a big one."""
    if n == 0:
        return 0.0
    p = k / n
    return (p + z * z / (2 * n) - z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)) / (1 + z * z / n)


def _filters(period) -> dict[str, Any]:
    return {"date_from": period.date_from, "date_to": period.date_to} if period is not None else {}


def _when(period, today: date) -> str:
    return " " + wording.period_phrase(period, today) if period is not None else ""


def _text(t: str) -> dict:
    return {"type": "text", "text": t}


def _pct(x: float) -> str:
    return f"{x:.0f}%"


def _main_complaint(db_path, product: str, filters: dict) -> tuple[str | None, str]:
    """(top negative topic, "its main problems: ..." sentence) for one product."""
    neg = [t for t in analytics.topic_breakdown(db_path, **{**filters, "product": product, "sentiment": "negative"}) if t["topic"] != "other"]
    if not neg:
        return None, ""
    top = neg[0]["topic"]
    rows, _ = database.fetch_reviews(sort="newest", page=1, page_size=500, db_path=db_path, **{**filters, "product": product, "sentiment": "negative", "topic": top})
    found, n = issues_service.common_issues(top, [r.get("review_text") or "" for r in rows], top=2)
    detail = ""
    if found:
        detail = " Typical problems: " + " and ".join(f"{label} ({c} of {n})" for label, c in found) + "."
    return top, detail


def _product_stats(db_path, period) -> list[dict]:
    stats = [s for s in analytics.by_product(db_path, **_filters(period)) if s["product"] != "(no product)"]
    for s in stats:
        n = s["positive"] + s["neutral"] + s["negative"]
        s["n"] = n
        s["pos_low"] = _wilson_low(s["positive"], n)
        s["neg_low"] = _wilson_low(s["negative"], n)
    return stats


def _table(stats: list[dict], db_path, filters: dict) -> dict:
    rows = []
    for s in stats:
        top, _ = _main_complaint(db_path, s["product"], filters)
        rows.append([s["product"], s["total"], _pct(s["positive_pct"]), _pct(s["negative_pct"]),
                     f"{s['avg_rating']:.2f}" if s["avg_rating"] is not None else "n/a", top or "none"])
    return {"type": "table", "columns": ["Product", "Reviews", "Positive %", "Negative %", "Avg rating", "Main complaint"], "rows": rows}


def recommend(kind: str, db_path, period, today: date) -> list[dict]:
    stats = _product_stats(db_path, period)
    when = _when(period, today)
    big = [s for s in stats if s["n"] >= _MIN_REVIEWS]
    if not big:
        return [_text(f"No product has at least {_MIN_REVIEWS} reviews{when}, so any recommendation would rest on too little. "
                      "Add more reviews or ask about a wider period.")]
    filters = _filters(period)
    if kind == "promote":
        ranked = sorted(big, key=lambda s: (-s["pos_low"], s["negative_pct"], -s["total"]))
        best, runner = ranked[0], ranked[1] if len(ranked) > 1 else None
        top, detail = _main_complaint(db_path, best["product"], filters)
        text = (f"**{best['product']}** looks like the safest product to promote{when}: {_pct(best['positive_pct'])} of its {best['n']} reviews are positive "
                f"and only {_pct(best['negative_pct'])} are negative" + (f", with an average rating of {best['avg_rating']:.2f} out of 5" if best["avg_rating"] is not None else "") + ".")
        if runner:
            text += f" Next best: **{runner['product']}** ({_pct(runner['positive_pct'])} positive of {runner['n']})."
        if top:
            text += f" Before you promote it, know its weak spot: **{top}**.{detail}"
        order = ranked
        note = "_I ranked products by how positive their reviews are, giving less credit to products with few reviews. Reviews show customer opinion only, not stock, margins or returns._"
    else:
        ranked = sorted(big, key=lambda s: (-s["neg_low"], -s["negative_pct"], -s["total"]))
        worst = ranked[0]
        most = max(big, key=lambda s: (s["negative"], s["total"]))
        top, detail = _main_complaint(db_path, worst["product"], filters)
        text = (f"**{worst['product']}** needs attention first{when}: {_pct(worst['negative_pct'])} of its {worst['n']} reviews are negative ({worst['negative']})"
                + (f" and its average rating is {worst['avg_rating']:.2f} out of 5" if worst["avg_rating"] is not None else "") + ".")
        if top:
            text += f" Its most common complaint is **{top}**.{detail}"
        if most["product"] != worst["product"]:
            text += f" **{most['product']}** has the most complaints in total ({most['negative']}), but that's {_pct(most['negative_pct'])} of a larger number of reviews."
        text += " Whether to stop selling something is a business decision: reviews show how customers feel, not profit or stock, so I'd fix the main complaint first and re-check."
        order = ranked
        note = "_Ranked by how negative a product's reviews are, giving less weight to products with few reviews._"
    blocks = [_text(text), _table(order[:8], db_path, filters), _text(note)]
    small = [s["product"] for s in stats if s["n"] < _MIN_REVIEWS]
    if small:
        blocks.append(_text(f"_Left out for having fewer than {_MIN_REVIEWS} reviews: {wording.join_names(small)}._"))
    return blocks


def mixed(db_path, period, today: date) -> list[dict]:
    stats = [s for s in _product_stats(db_path, period) if s["n"] >= 5]
    when = _when(period, today)
    if not stats:
        return [_text(f"No product has enough reviews{when} to judge. I need at least 5 per product.")]
    for s in stats:
        s["mix"] = min(s["positive_pct"], s["negative_pct"])
    stats.sort(key=lambda s: (-s["mix"], -s["total"]))
    top = stats[0]
    if top["mix"] < 15:
        head = f"No product has clearly mixed reviews{when}: each one leans mostly one way."
    else:
        head = (f"**{top['product']}** has the most mixed feedback{when}: {_pct(top['positive_pct'])} of its {top['n']} reviews are positive and "
                f"{_pct(top['negative_pct'])} are negative.")
        others = [s["product"] for s in stats[1:3] if s["mix"] >= 15]
        if others:
            head += f" Also fairly split: {wording.join_names(others)}."
    table = {"type": "table", "columns": ["Product", "Reviews", "Positive %", "Neutral %", "Negative %"],
             "rows": [[s["product"], s["total"], _pct(s["positive_pct"]), _pct(100 - s["positive_pct"] - s["negative_pct"]), _pct(s["negative_pct"])] for s in stats[:8]]}
    return [_text(head), table, _text("_“Mixed” here means a product has both a large share of positive and a large share of negative reviews._")]


def compare_topics(a: str, b: str, negative: bool, db_path, period, today: date) -> list[dict]:
    when = _when(period, today)
    by_topic = {t["topic"]: t for t in analytics.topic_breakdown(db_path, **_filters(period))}
    empty = {"total": 0, "positive": 0, "neutral": 0, "negative": 0, "negative_pct": 0.0}
    ta, tb = by_topic.get(a, {**empty, "topic": a}), by_topic.get(b, {**empty, "topic": b})
    metric, noun = ("negative", "complaints (negative reviews)") if negative else ("total", "reviews")
    if ta[metric] == 0 and tb[metric] == 0:
        return [_text(f"No reviews about {a} or {b}{when}." + _coverage(db_path))]
    if ta[metric] == tb[metric]:
        head = f"{a.capitalize()} and {b} have the same number of {noun}{when}: {ta[metric]} each."
    else:
        hi, lo = (ta, tb) if ta[metric] > tb[metric] else (tb, ta)
        head = f"**{hi['topic'].capitalize()}** has more {noun} than **{lo['topic']}**{when}: {hi[metric]} against {lo[metric]}."
        if negative:
            head += (f" Measured as a share of the reviews that mention each topic, it's {_pct(hi['negative_pct'])} for {hi['topic']} and "
                     f"{_pct(lo['negative_pct'])} for {lo['topic']}.")
    table = {"type": "table", "columns": ["Topic", "Reviews mentioning it", "Positive", "Neutral", "Negative", "Negative %"],
             "rows": [[t["topic"], t["total"], t["positive"], t["neutral"], t["negative"], _pct(t["negative_pct"])] for t in (ta, tb)]}
    return [_text(head), table, _text("_A review can mention more than one topic, so the rows overlap._")]


def both_topics(a: str, b: str, negative: bool, db_path, period, today: date) -> tuple[list[dict], list[dict]]:
    """Of the reviews that mention topic A (negative ones for a complaint question), how many also mention B?"""
    when = _when(period, today)
    sentiment = "negative" if negative else None
    base = {**_filters(period), **({"sentiment": sentiment} if sentiment else {})}
    rows, _ = database.fetch_reviews(sort="newest", page=1, page_size=1000, db_path=db_path, **{**base, "topic": a})
    tagged = [r for r in rows if b in topics_service.from_storage(r.get("topics"))]
    everyone, _ = database.fetch_reviews(sort="newest", page=1, page_size=1000, db_path=db_path, **base)
    base_n = sum(1 for r in everyone if b in topics_service.from_storage(r.get("topics")))
    kind = "negative reviews" if negative else "reviews"
    if not rows:
        return [_text(f"There are no {kind} about {a}{when}." + _coverage(db_path))], []
    share = len(tagged) / len(rows)
    base_share = base_n / max(len(everyone), 1)
    head = (f"{'Some do' if tagged else 'No'}: {len(tagged)} of the {len(rows)} {kind} about {a}{when} ({_pct(100 * share)}) also mention {b}.")
    if tagged:
        head = f"{'Only a few do' if share < 0.25 else 'Some do' if share < 0.5 else 'Many do'}: {len(tagged)} of the {len(rows)} {kind} about {a}{when} ({_pct(100 * share)}) also mention {b}."
    if everyone:
        compare = "more" if share > base_share + 0.05 else "less" if share < base_share - 0.05 else "about as"
        head += (f" Across all {kind}{when} {_pct(100 * base_share)} mention {b}, so people who raise {a} are {compare} likely to mention it"
                 + ("." if compare != "about as" else " as everyone else."))
    return [_text(head)], tagged[:4]


def _coverage(db_path) -> str:
    first, last = database.get_date_range(db_path)
    return f" Your reviews run from {first[:10]} to {last[:10]}." if first and last else ""
