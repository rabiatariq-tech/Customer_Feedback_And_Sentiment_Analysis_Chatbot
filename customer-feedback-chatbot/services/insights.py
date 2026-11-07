"""Plain-English readings of the numbers: trends, topic details, rough fractions.

Every sentence here is built from database counts, never invented. The same facts can later be handed to a
language model to phrase more fluently, but they never depend on one.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import database
from services import analytics, topics as topic_service

MIN_REVIEWS_FOR_TREND = 5   # per period, otherwise we say there is too little data
CHANGE_POINTS = 10          # percentage-point change that counts as "rising" or "improving"


def one_in(part: int, whole: int) -> str:
    """'about 1 in 3' style phrase for a share."""
    if whole <= 0 or part <= 0:
        return "none"
    share = part / whole
    if share >= 0.9:
        return "nearly all"
    if share >= 0.65:
        return "most"
    if share >= 0.45:
        return "about half"
    for n in (3, 4, 5, 6, 8, 10):
        if share >= 1 / n * 0.85:
            return f"about 1 in {n}"
    return "fewer than 1 in 10"


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
    return {"recent": recent, "earlier": earlier, "days": days, "cut": cut, "last": last}


def describe_trend(label: str, r: dict[str, Any] | None) -> str:
    """Two short sentences: what happened recently versus before, and whether that is a real change."""
    if r is None:
        return "There is too little data over time to say whether things are changing."
    cur, prev = r["recent"], r["earlier"]
    cn, pn = cur["negative"], prev["negative"]
    ct = cur["positive"] + cur["neutral"] + cur["negative"]
    pt = prev["positive"] + prev["neutral"] + prev["negative"]
    when = f"the last {window_text(r['days'])}"
    if ct == 0 or pt == 0:
        return f"In {when} there were {cur['total']} {label}, and {prev['total']} before that, so I can't compare them."
    cs, ps = cn / ct, pn / pt
    sentence = (f"In {when}, {cn} of {ct} {label} ({cs:.0%}) were negative, "
                f"compared with {pn} of {pt} ({ps:.0%}) before that.")
    if min(ct, pt) < MIN_REVIEWS_FOR_TREND:
        return sentence + " One of those periods has very few reviews, so this is too small to call a trend."
    diff = (cs - ps) * 100
    if diff >= CHANGE_POINTS:
        return sentence + " **Complaints are rising.**"
    if diff <= -CHANGE_POINTS:
        return sentence + " **Complaints are falling.**"
    return sentence + " That is no clear change."


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
