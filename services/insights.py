"""Plain-English readings of the numbers: trends, topic details, rough fractions.

Every sentence here is built from database counts, never invented. The same facts can later be handed to a
language model to phrase more fluently, but they never depend on one.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import database
from services import analytics, topics as topic_service, wording

MIN_REVIEWS_FOR_TREND = 5   # per period, otherwise we say there is too little data
MAX_CHANCE = 0.10           # a change counts as real only if it would happen by luck less than 1 time in 10
CHANGE_POINTS = 10          # percentage-point change that counts as "rising" or "improving"


def one_in(part: int, whole: int) -> str:
    """A share in words ("about a third", "most", "nearly all"). Kept under this name for older callers."""
    return wording.proportion(part, whole)


def windows(first_day: str, last_day: str) -> tuple[date, date, date, int] | None:
    """(first, cut, last, window_days): recent = cut..last, earlier = first..cut-1. None if the data is too short."""
    first, last = date.fromisoformat(first_day), date.fromisoformat(last_day)
    span = (last - first).days + 1
    if span < 14:
        return None
    window = 28 if span >= 56 else max(7, span // 3)
    return first, last - timedelta(days=window - 1), last, window


def window_text(days: int) -> str:
    return f"{days // 7} weeks" if days % 7 == 0 else f"{days} days"


def recent_vs_earlier(db_path, filters: dict[str, Any]) -> dict[str, Any] | None:
    """Overview numbers for the recent window and the time before it, for the same filters."""
    base = analytics.overview(db_path, **filters)
    if base["total"] < 8 or not base["first_day"]:
        return None
    w = windows(base["first_day"], base["last_day"])
    if not w:
        return None
    first, cut, last, days = w
    recent = analytics.overview(db_path, **{**filters, "date_from": cut.isoformat(), "date_to": last.isoformat()})
    earlier = analytics.overview(db_path, **{**filters, "date_from": first.isoformat(),
                                             "date_to": (cut - timedelta(days=1)).isoformat()})
    return {"recent": recent, "earlier": earlier, "days": days, "cut": cut, "last": last, "first": first}


def chance_of_difference(neg_a: int, total_a: int, neg_b: int, total_b: int) -> float:
    """Fisher's exact test (two-sided): how likely a gap at least this big is by pure chance if nothing really changed.
    Small is convincing (0.05 = about 1 time in 20); large means the gap could easily be luck."""
    from math import comb

    n, row, col = total_a + total_b, total_a, neg_a + neg_b
    def prob(x: int) -> float:
        return comb(row, x) * comb(n - row, col - x) / comb(n, col)
    observed = prob(neg_a)
    return min(1.0, sum(prob(x) for x in range(max(0, col - (n - row)), min(row, col) + 1) if prob(x) <= observed * (1 + 1e-9)))


def _day(d) -> str:
    return f"{d:%b} {d.day}"


def describe_trend(label: str, r: dict[str, Any] | None) -> str:
    """A one-word verdict first, then the two periods side by side in plain words, then why we are or are not sure."""
    if r is None:
        return "There isn't enough data over time yet to say whether things are changing."
    cur, prev = r["recent"], r["earlier"]
    cn, pn = cur["negative"], prev["negative"]
    ct = cur["positive"] + cur["neutral"] + cur["negative"]
    pt = prev["positive"] + prev["neutral"] + prev["negative"]
    when = f"the last {window_text(r['days'])}"
    if ct == 0 or pt == 0:
        return f"There {'was' if cur['total'] == 1 else 'were'} {cur['total']} {label} in {when} and {prev['total']} before that, so I can't compare them."
    cs, ps = cn / ct, pn / pt
    recent_range = f" ({_day(r['cut'])} – {_day(r['last'])})" if r.get("cut") and r.get("last") else ""
    first = r.get("first")
    earlier_range = f" ({_day(first)} – {_day(r['cut'] - timedelta(days=1))})" if first and r.get("cut") else ""
    lines = (f"- **Last {window_text(r['days'])}**{recent_range}: {cn} of {ct} {label} were negative ({cs:.0%}).\n"
             f"- **Before that**{earlier_range}: {pn} of {pt} were negative ({ps:.0%}).")

    def head(verdict: str, why: str) -> str:
        return f"**{verdict}** {why}\n\n{lines}"

    if min(ct, pt) < MIN_REVIEWS_FOR_TREND:
        return head("Too early to tell.", "One of the two periods has very few reviews, so the comparison isn't reliable yet.")
    diff = (cs - ps) * 100
    if abs(diff) >= CHANGE_POINTS:
        if chance_of_difference(cn, ct, pn, pt) > MAX_CHANCE:  # a big-looking gap from a handful of reviews is often luck
            small = min(ct, pt)
            return (head("Not clear yet.", f"The share of negative reviews {'went up' if diff > 0 else 'went down'}, but the gap is small compared with how few reviews there are.")
                    + f"\n\nWith only {small} reviews in the smaller period, a gap like this could easily happen by chance. "
                      "I'd wait for more reviews before calling it a trend.")
        return head("Rising." if diff > 0 else "Falling.",
                    "That's a real change, not just chance: **complaints are rising.**" if diff > 0
                    else "That's a real change, not just chance: **complaints are falling.**")
    return head("No clear change.", "The share of negative reviews is about the same in both periods.")


def topic_details(db_path, filters: dict[str, Any], focus: str, names: list[str]) -> dict[str, dict[str, Any]]:
    """For each topic among the focus-sentiment reviews: how many, which product has most of them, how many are recent."""
    rows, _ = database.fetch_reviews(sort="newest", page=1, page_size=500, db_path=db_path, sentiment=focus, **filters)
    base = analytics.overview(db_path, **{k: v for k, v in filters.items()})
    w = windows(base["first_day"], base["last_day"]) if base["total"] and base["first_day"] else None
    out: dict[str, dict[str, Any]] = {}
    for name in names:
        mine = [r for r in rows if name in topic_service.from_storage(r.get("topics"))]
        by_product: dict[str, int] = {}
        for r in mine:
            if r.get("product"):
                by_product[r["product"]] = by_product.get(r["product"], 0) + 1
        top_product = max(by_product.items(), key=lambda kv: (kv[1], kv[0])) if by_product else None
        recent = 0
        rising = False
        if w:
            first, cut, last, days = w
            recent = sum(1 for r in mine if str(r.get("created_at") or "")[:10] >= cut.isoformat())
            expected = len(mine) * days / ((last - first).days + 1)
            rising = len(mine) >= 6 and recent >= 1.5 * expected and recent - expected >= 2
        out[name] = {"count": len(mine), "top_product": top_product, "recent": recent,
                     "window": window_text(w[3]) if w else None, "rising": rising}
    return out
