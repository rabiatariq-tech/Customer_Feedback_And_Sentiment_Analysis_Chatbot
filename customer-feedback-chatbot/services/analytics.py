"""Read-only analytics over the feedback table. Every number comes from SQL or plain Python,
never from an LLM, so the chatbot can quote them as facts.

All functions accept the same optional filters as ``database.fetch_reviews``:
date_from, date_to, product, sentiment, search, extra_key, extra_value, flag, batch_id, topic.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import config
import database
from services import topics as topic_service

_FILTERS = (
    "date_from", "date_to", "product", "sentiment", "search",
    "extra_key", "extra_value", "flag", "batch_id", "topic", "exclude_topics",
)
_PERIODS = {"day": "%Y-%m-%d", "week": "%Y-W%W", "month": "%Y-%m"}


def _where(filters: dict[str, Any]) -> tuple[str, list[Any]]:
    unknown = set(filters) - set(_FILTERS)
    if unknown:
        raise TypeError(f"unknown filter(s): {', '.join(sorted(unknown))}")
    return database.build_where(*(filters.get(name) for name in _FILTERS))


def _pct(part: int, whole: int) -> float:
    return round(100 * part / whole, 1) if whole else 0.0


def overview(db_path: str | Path | None = None, **filters: Any) -> dict[str, Any]:
    """Totals, sentiment split, average rating/confidence and number of low-confidence reviews."""
    where, params = _where(filters)
    with database.get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS total, "
            "SUM(sentiment_label='positive') AS pos, SUM(sentiment_label='neutral') AS neu, "
            "SUM(sentiment_label='negative') AS neg, SUM(sentiment_label IS NULL) AS pending, "
            "AVG(rating) AS avg_rating, "
            "AVG(CASE sentiment_label WHEN 'positive' THEN score_positive WHEN 'neutral' THEN score_neutral "
            "WHEN 'negative' THEN score_negative END) AS avg_conf, "
            "SUM(instr(',' || COALESCE(flags, '') || ',', ',low_confidence,') > 0) AS low_conf, "
            "MIN(date(created_at)) AS first_day, MAX(date(created_at)) AS last_day "
            f"FROM feedback {where}",
            params,
        ).fetchone()
    total = int(row["total"] or 0)
    pos, neu, neg = (int(row[k] or 0) for k in ("pos", "neu", "neg"))
    analysed = pos + neu + neg
    return {
        "total": total,
        "positive": pos, "neutral": neu, "negative": neg,
        "pending": int(row["pending"] or 0),
        "positive_pct": _pct(pos, analysed), "neutral_pct": _pct(neu, analysed), "negative_pct": _pct(neg, analysed),
        "avg_rating": round(row["avg_rating"], 2) if row["avg_rating"] is not None else None,
        "avg_confidence": round(row["avg_conf"], 3) if row["avg_conf"] is not None else None,
        "low_confidence": int(row["low_conf"] or 0),
        "first_day": row["first_day"], "last_day": row["last_day"],
    }


def sentiment_trend(granularity: str = "day", db_path: str | Path | None = None, **filters: Any) -> list[dict[str, Any]]:
    """Reviews per period split by sentiment, oldest first. granularity: day, week or month."""
    if granularity not in _PERIODS:
        raise ValueError(f"granularity must be one of {', '.join(_PERIODS)}")
    where, params = _where(filters)
    with database.get_connection(db_path) as conn:
        rows = conn.execute(
            f"SELECT strftime('{_PERIODS[granularity]}', created_at) AS period, COUNT(*) AS total, "
            "SUM(sentiment_label='positive') AS pos, SUM(sentiment_label='neutral') AS neu, "
            "SUM(sentiment_label='negative') AS neg "
            f"FROM feedback {where} GROUP BY period ORDER BY period",
            params,
        ).fetchall()
    return [
        {"period": r["period"], "total": int(r["total"]), "positive": int(r["pos"] or 0),
         "neutral": int(r["neu"] or 0), "negative": int(r["neg"] or 0)}
        for r in rows
    ]


def by_product(db_path: str | Path | None = None, **filters: Any) -> list[dict[str, Any]]:
    """Per product: counts, negative share and average rating. Most negative share first."""
    where, params = _where(filters)
    with database.get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT COALESCE(product, '(no product)') AS product, COUNT(*) AS total, "
            "SUM(sentiment_label='positive') AS pos, SUM(sentiment_label='neutral') AS neu, "
            "SUM(sentiment_label='negative') AS neg, AVG(rating) AS avg_rating, COUNT(rating) AS rated "
            f"FROM feedback {where} GROUP BY product",
            params,
        ).fetchall()
    out = []
    for r in rows:
        pos, neu, neg = (int(r[k] or 0) for k in ("pos", "neu", "neg"))
        out.append({
            "product": r["product"], "total": int(r["total"]), "positive": pos, "neutral": neu, "negative": neg,
            "negative_pct": _pct(neg, pos + neu + neg), "positive_pct": _pct(pos, pos + neu + neg),
            "rated": int(r["rated"] or 0),
            "avg_rating": round(r["avg_rating"], 2) if r["avg_rating"] is not None else None,
        })
    out.sort(key=lambda d: (-d["negative_pct"], -d["total"], d["product"]))
    return out


def topic_breakdown(db_path: str | Path | None = None, **filters: Any) -> list[dict[str, Any]]:
    """Per topic: mentions and sentiment split. Most negative mentions first. A review can count in several topics."""
    where, params = _where(filters)
    with database.get_connection(db_path) as conn:
        rows = conn.execute(
            f"SELECT topics, sentiment_label FROM feedback {where}", params
        ).fetchall()
    stats: dict[str, dict[str, int]] = {}
    for r in rows:
        for name in topic_service.from_storage(r["topics"]):
            s = stats.setdefault(name, {"total": 0, "positive": 0, "neutral": 0, "negative": 0})
            s["total"] += 1
            if r["sentiment_label"] in ("positive", "neutral", "negative"):
                s[r["sentiment_label"]] += 1
    out = [
        {"topic": name, **s, "negative_pct": _pct(s["negative"], s["positive"] + s["neutral"] + s["negative"])}
        for name, s in stats.items()
    ]
    out.sort(key=lambda d: (-d["negative"], -d["total"], d["topic"]))
    return out


def examples(limit: int = 5, db_path: str | Path | None = None, **filters: Any) -> list[dict[str, Any]]:
    """Evidence: up to ``limit`` of the newest matching reviews."""
    rows, _ = database.fetch_reviews(page=1, page_size=max(1, min(limit, 50)), sort="newest", db_path=db_path, **filters)
    return rows


def compare_periods(
    date_from: str, date_to: str, db_path: str | Path | None = None, **filters: Any
) -> dict[str, Any]:
    """Compare [date_from, date_to] with the equally long period right before it."""
    start, end = date.fromisoformat(date_from), date.fromisoformat(date_to)
    if end < start:
        raise ValueError("date_to must not be before date_from")
    length = (end - start).days + 1
    prev_end = start - timedelta(days=1)
    prev_start = prev_end - timedelta(days=length - 1)
    current = overview(db_path, date_from=start.isoformat(), date_to=end.isoformat(), **filters)
    previous = overview(db_path, date_from=prev_start.isoformat(), date_to=prev_end.isoformat(), **filters)
    return {
        "current": current,
        "previous": previous,
        "previous_range": (prev_start.isoformat(), prev_end.isoformat()),
        "negative_pct_change": round(current["negative_pct"] - previous["negative_pct"], 1),
        "total_change": current["total"] - previous["total"],
        "enough_data": min(current["total"], previous["total"]) >= 5,
    }


def choose_granularity(days_span: int) -> str:
    """Bucket size for a time chart: short ranges by day, medium by week, long by month."""
    if days_span <= 45:
        return "day"
    if days_span <= 200:
        return "week"
    return "month"


def period_label(period: str, granularity: str) -> str:
    """Sortable, readable label for a trend bucket: day -> 2026-10-05, week -> its Monday, month -> 2026-10."""
    if granularity == "week":
        try:
            return datetime.strptime(period + "-1", "%Y-W%W-%w").date().isoformat()
        except ValueError:
            return period
    return period


def fill_trend(
    rows: list[dict[str, Any]], granularity: str, date_from: str | None, date_to: str | None
) -> list[dict[str, Any]]:
    """Add zero rows for periods without reviews so the time axis has no gaps. Needs a start and end date."""
    if not rows and not (date_from and date_to):
        return rows
    try:
        start = date.fromisoformat(date_from) if date_from else None
        end = date.fromisoformat(date_to) if date_to else None
    except ValueError:
        return rows
    if start is None or end is None:
        return rows
    if granularity == "week":
        start = start - timedelta(days=start.weekday())
    elif granularity == "month":
        start = start.replace(day=1)
    existing = {r["period"]: r for r in rows}
    out, cursor, guard = [], start, 0
    while cursor <= end and guard < 800:
        key = cursor.strftime(_PERIODS[granularity])
        out.append(existing.pop(key, {"period": key, "total": 0, "positive": 0, "neutral": 0, "negative": 0}))
        if granularity == "day":
            cursor += timedelta(days=1)
        elif granularity == "week":
            cursor += timedelta(days=7)
        else:
            cursor = (cursor.replace(day=28) + timedelta(days=4)).replace(day=1)
        guard += 1
    out.extend(existing.values())  # anything outside the range stays visible
    return sorted(out, key=lambda r: r["period"])
