"""Which products moved when the overall picture changed ("which product caused the rise in positive reviews?").

Pure arithmetic on the per-product counts that ``analytics.by_product`` already returns for two periods. Nothing here
claims a cause: reviews show WHAT changed per product, never WHY, so the wording says "accounts for" a share of the
change in review counts, and small samples are flagged instead of ranked as if they were reliable.
"""
from __future__ import annotations

import calendar
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from services import analytics

MIN_REVIEWS = 5          # fewer reviews than this in either period: too few to read a change into
NO_PRODUCT = "(no product)"


def previous_window(date_from: str, date_to: str) -> tuple[str, str]:
    """The period to compare with: the previous calendar month for a whole month, else the equally long period before."""
    start, end = date.fromisoformat(date_from), date.fromisoformat(date_to)
    if start.day == 1 and end == date(start.year, start.month, calendar.monthrange(start.year, start.month)[1]):
        last = start - timedelta(days=1)
        return last.replace(day=1).isoformat(), last.isoformat()
    if start.day == 1 and end.month == start.month and end.year == start.year:
        # a month so far ("this month" on the 21st): compare with the same days of the previous month, not a shifted window
        last = start - timedelta(days=1)
        return last.replace(day=1).isoformat(), last.replace(day=min(end.day, last.day)).isoformat()
    length = (end - start).days + 1
    prev_end = start - timedelta(days=1)
    return (prev_end - timedelta(days=length - 1)).isoformat(), prev_end.isoformat()


def product_changes(db_path: str | Path | None, current: tuple[str, str], previous: tuple[str, str], **filters: Any) -> dict[str, Any]:
    """Per-product counts in both periods, the change, and the overall totals (which are the sum of the rows, so the
    per-product changes add up exactly to the overall change)."""
    filters = {k: v for k, v in filters.items() if k not in ("date_from", "date_to", "product", "sentiment")}
    cur = {s["product"]: s for s in analytics.by_product(db_path, date_from=current[0], date_to=current[1], **filters)}
    prev = {s["product"]: s for s in analytics.by_product(db_path, date_from=previous[0], date_to=previous[1], **filters)}
    zero = {"total": 0, "positive": 0, "neutral": 0, "negative": 0, "negative_pct": 0.0, "positive_pct": 0.0}
    rows = []
    for name in sorted(set(cur) | set(prev)):
        c, p = cur.get(name, zero), prev.get(name, zero)
        rows.append({
            "product": name,
            "total": (p["total"], c["total"]),
            "positive": (p["positive"], c["positive"]), "negative": (p["negative"], c["negative"]),
            "d_positive": c["positive"] - p["positive"], "d_negative": c["negative"] - p["negative"],
            "negative_pct": (p["negative_pct"], c["negative_pct"]),
            "few": min(c["total"], p["total"]) < MIN_REVIEWS,
        })
    totals = {k: (sum(r[k][0] for r in rows), sum(r[k][1] for r in rows)) for k in ("total", "positive", "negative")}
    return {"rows": rows, "overall": totals, "current": current, "previous": previous}


def _movers(rows: list[dict[str, Any]], key: str, overall_delta: int, limit: int = 3) -> list[dict[str, Any]]:
    """Named products whose change goes the same way as the overall change, biggest first. Products with too few
    reviews are not named as drivers (their change may be noise)."""
    sign = 1 if overall_delta > 0 else -1
    picked = [r for r in rows if r["product"] != NO_PRODUCT and not r["few"] and r[key] * sign > 0]
    return sorted(picked, key=lambda r: (-abs(r[key]), r["product"]))[:limit]


def _signed(n: int) -> str:
    return f"+{n}" if n > 0 else f"−{abs(n)}" if n < 0 else "0"


def _sentence(result: dict[str, Any], key: str, word: str) -> str:
    before, after = result["overall"][key]
    delta = after - before
    head = (f"{word.capitalize()} reviews {'rose' if delta > 0 else 'fell' if delta < 0 else 'stayed at'} "
            + (f"from {before} to {after} ({_signed(delta)})" if delta else f"{after}"))
    if not delta:
        return head + "."
    delta_key = "d_" + key
    drivers = _movers(result["rows"], delta_key, delta)
    if drivers:
        names = ", ".join(f"**{r['product']}** ({_signed(r[delta_key])})" for r in drivers)
        head += f". The {'rise' if delta > 0 else 'drop'} is mostly accounted for by {names}"
    else:
        head += ". No single product with enough reviews stands out"
    against = [r for r in result["rows"] if r["product"] != NO_PRODUCT and not r["few"] and r[delta_key] * delta < 0]
    if against:
        worst = sorted(against, key=lambda r: (-abs(r[delta_key]), r["product"]))[0]
        head += f"; the other way, **{worst['product']}** went {_signed(worst[delta_key])}"
    return head + "."


def describe(result: dict[str, Any], focus: str | None = None) -> list[str]:
    """Plain-English sentences: one for positive and one for negative reviews (or only `focus` when asked about one)."""
    parts = []
    for key, word in (("positive", "positive"), ("negative", "negative")):
        if focus in (None, key):
            parts.append(_sentence(result, key, word))
    before, after = result["overall"]["total"]
    if before and abs(after - before) / before >= 0.25:
        parts.append(f"Total reviews went from {before} to {after}, so part of any change in counts is just volume; "
                     "the negative-share column shows the rate.")
    return parts


def table_rows(result: dict[str, Any]) -> list[list[Any]]:
    """Rows for the chat table, biggest overall movement first; the 'no product' bucket last so the sums still add up."""
    named = [r for r in result["rows"] if r["product"] != NO_PRODUCT]
    named.sort(key=lambda r: (-(abs(r["d_positive"]) + abs(r["d_negative"])), r["product"]))
    ordered = named + [r for r in result["rows"] if r["product"] == NO_PRODUCT]
    out = []
    for r in ordered:
        out.append([
            r["product"] + (" (few reviews)" if r["few"] else ""),
            f"{r['total'][0]} → {r['total'][1]}",
            f"{r['positive'][0]} → {r['positive'][1]} ({_signed(r['d_positive'])})",
            f"{r['negative'][0]} → {r['negative'][1]} ({_signed(r['d_negative'])})",
            f"{r['negative_pct'][0]:.0f}% → {r['negative_pct'][1]:.0f}%",
        ])
    return out


TABLE_COLUMNS = ["Product", "Reviews", "Positive", "Negative", "Negative %"]
