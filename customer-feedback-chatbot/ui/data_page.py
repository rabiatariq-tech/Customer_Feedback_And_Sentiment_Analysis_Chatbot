"""Data view: filters, metric cards, charts and a paginated table over the same database as the chat."""
from __future__ import annotations

import json
from datetime import date, timedelta

import pandas as pd
import streamlit as st

import config
import database
from services import analytics, topics
from ui import nav, session, theme

SENT_COLORS = [theme.SENT["positive"][0], theme.SENT["neutral"][0], theme.SENT["negative"][0]]
SENTIMENTS = ("positive", "neutral", "negative")
FLAG_LABELS = {"All": None, "Only unsure ones": "low_confidence"}
FLAG_WORDS = {"low_confidence": "unsure", "truncated": "text cut off"}


def _select(label: str, key: str, options: list[str], preferred: str | None, container) -> str:
    """Selectbox whose value survives page switches (state is rebuilt from `dv_pref`)."""
    ss = st.session_state
    if key not in ss:
        ss[key] = preferred if preferred in options else "All"
    elif ss[key] not in options:
        ss[key] = "All"
    return container.selectbox(label, options, key=key)


def _to_date(value: str | None, fallback: date) -> date:
    try:
        return date.fromisoformat(value) if value else fallback
    except ValueError:
        return fallback


def _table_frame(rows: list[dict]) -> pd.DataFrame:
    """Rows from the database as a display table; custom CSV columns become extra columns."""
    records = []
    for r in rows:
        label = r.get("sentiment_label")
        record = {
            "id": r.get("id"),
            "created_at": str(r.get("created_at") or "")[:10],
            "product": r.get("product") or "",
            "sentiment": label or "pending",
            "confidence": r.get(f"score_{label}") if label else None,
            "rating": r.get("rating"),
            "topics": ", ".join(topics.from_storage(r.get("topics"))),
            "source": r.get("source") or "",
            "note": ", ".join(FLAG_WORDS.get(f, f) for f in (r.get("flags") or "").split(",") if f),
            "text": r.get("review_text") or "",
        }
        try:
            extra = json.loads(r["extra"]) if r.get("extra") else {}
        except (TypeError, ValueError):
            extra = {}
        for key, value in extra.items():
            record.setdefault(key, value)
        records.append(record)
    first = ["id", "created_at", "product", "sentiment", "confidence", "rating", "topics", "note", "text", "source"]
    frame = pd.DataFrame(records, columns=None if records else first)
    return frame[[c for c in first if c in frame.columns] + [c for c in frame.columns if c not in first]]


def render() -> None:
    theme.inject(1180)
    ss = st.session_state
    db = session.db_path()
    pref: dict = ss.get("dv_pref", {})

    head_l, head_r = st.columns([5, 1.4])
    head_l.markdown(
        '<div class="page-title">Data view</div>'
        '<div class="page-sub">The same reviews you chat about, with filters, charts and a table.</div>',
        unsafe_allow_html=True,
    )
    if head_r.button("Back to chat", icon=":material/arrow_back:", width="stretch"):
        st.switch_page(nav.CHAT)

    total_reviews = database.count_reviews(db)
    if total_reviews == 0:
        st.info("There are no reviews yet. Go back to the chat and attach a CSV.")
        return

    # ---- filters ----------------------------------------------------------
    first_day, last_day = database.get_date_range(db)
    min_d = _to_date(first_day, date.today())
    max_d = _to_date(last_day, date.today())
    if "dv_w_date" not in ss:
        start, end = pref.get("date_range", (min_d, max_d))
        if pref.get("date_from") and pref.get("date_to"):
            start, end = _to_date(pref["date_from"], min_d), _to_date(pref["date_to"], max_d)
        if "days" in pref:
            end = max_d
            start = max_d - timedelta(days=int(pref["days"]) - 1)
        ss["dv_w_date"] = (max(min_d, min(start, max_d)), min(max_d, max(end, min_d)))
    else:
        cur = ss["dv_w_date"]
        if not isinstance(cur, (tuple, list)) or any(d < min_d or d > max_d for d in cur):
            ss["dv_w_date"] = (min_d, max_d)

    extra_keys = [k["key"] for k in database.get_extra_keys(db) if 1 < k["distinct_values"] <= 30]

    with st.container(border=True):
        r1 = st.columns([2, 1.4, 1.2, 1.4])
        date_value = r1[0].date_input("Date range", min_value=min_d, max_value=max_d, key="dv_w_date")
        product = _select("Product", "dv_w_product", ["All"] + database.get_products(db), pref.get("product"), r1[1])
        sentiment = _select("Sentiment", "dv_w_sentiment", ["All", *SENTIMENTS, "unanalyzed"], pref.get("sentiment"), r1[2])
        topic = _select("Topic", "dv_w_topic", ["All", *topics.TOPICS], pref.get("topic"), r1[3])
        r2 = st.columns([2, 1.4, 1.2, 1.4])
        if "dv_w_search" not in ss:
            ss["dv_w_search"] = pref.get("search") or ""
        search = r2[0].text_input("Search reviews and products", key="dv_w_search", placeholder="e.g. refund, delivery (comma = or)")
        extra_name = _select("Extra column", "dv_w_extra", ["All", *extra_keys], pref.get("extra"), r2[1])
        extra_value = "All"
        if extra_name != "All":
            values = [v for v, _ in database.get_extra_values(extra_name, limit=100, db_path=db)]
            extra_value = _select(f"{extra_name} value", "dv_w_extra_val", ["All", *values], pref.get("extra_val"), r2[2])
        elif "dv_w_extra_val" in ss:
            del ss["dv_w_extra_val"]
        flag_label = _select("Model certainty", "dv_w_flag", list(FLAG_LABELS), pref.get("flag"), r2[3])
        r2[3].caption(f"“Unsure” = the model's best guess scored under {config.SENTIMENT_LOW_CONFIDENCE:.0%}.")

    d1 = date_value[0] if isinstance(date_value, (tuple, list)) and date_value else date_value
    d2 = date_value[1] if isinstance(date_value, (tuple, list)) and len(date_value) > 1 else max_d
    filters = {
        "date_from": d1.isoformat(),
        "date_to": d2.isoformat(),
        "product": None if product == "All" else product,
        "sentiment": None if sentiment == "All" else sentiment,
        "topic": None if topic == "All" else topic,
        "search": [t.strip() for t in search.split(",") if t.strip()] or None,
        "extra_key": None if extra_name == "All" or extra_value == "All" else extra_name,
        "extra_value": None if extra_name == "All" or extra_value == "All" else extra_value,
        "flag": FLAG_LABELS[flag_label],
    }

    ss["dv_pref"] = {
        "date_range": (d1, d2), "product": product, "sentiment": sentiment, "topic": topic,
        "extra": extra_name, "extra_val": extra_value, "flag": flag_label, "search": search,
    }

    o = analytics.overview(db, **filters)
    st.caption(f"{o['total']:,} of {total_reviews:,} reviews match the filters.")
    if o["total"] == 0:
        st.info("No reviews match these filters. Try widening the date range or clearing the search.")
        return

    # ---- metrics ----------------------------------------------------------
    analysed = o["positive"] + o["neutral"] + o["negative"]
    base = max(analysed, 1)
    avg = o["avg_rating"]
    cards = [
        ("Reviews", f"{o['total']:,}", f"{o['pending']:,} not analysed yet" if o["pending"] else "after filters"),
        ("Positive", f"{o['positive'] / base:.0%}", f"{o['positive']:,} reviews"),
        ("Neutral", f"{o['neutral'] / base:.0%}", f"{o['neutral']:,} reviews"),
        ("Negative", f"{o['negative'] / base:.0%}", f"{o['negative']:,} reviews"),
        ("Average rating", f"{avg:.1f} / 5" if avg is not None else "n/a",
         "reviews with a rating" if avg is not None else "no ratings in this selection"),
    ]
    for col, (label, value, hint) in zip(st.columns(5), cards):
        col.markdown(theme.metric_card(label, value, hint), unsafe_allow_html=True)
    st.write("")

    # ---- charts -----------------------------------------------------------
    left, right = st.columns([1, 2], gap="large")
    with left:
        st.markdown("**Sentiment mix**")
        st.markdown(theme.bars({k: o[k] for k in SENTIMENTS}), unsafe_allow_html=True)
        st.markdown("**Negative reviews by topic**")
        neg_filters = {**filters, "sentiment": "negative"}
        neg_topics = analytics.topic_breakdown(db, **neg_filters)
        if not neg_topics:
            st.caption("No negative reviews in this selection.")
        else:
            series = pd.Series({t["topic"]: t["negative"] for t in neg_topics})
            st.bar_chart(series, color=theme.SENT["negative"][0], horizontal=True, height=200)
    with right:
        gran = analytics.choose_granularity((d2 - d1).days)
        st.markdown(f"**Reviews over time** · per {gran}")
        trend = analytics.sentiment_trend(gran, db, **filters)
        if trend:
            trend = analytics.fill_trend(trend, gran, d1.isoformat(), d2.isoformat())
            frame = pd.DataFrame(trend)
            frame["label"] = [analytics.period_label(p, gran) for p in frame["period"]]
            frame = frame.set_index("label")[list(SENTIMENTS)]
            x_label = {"day": "Day", "week": "Week starting (Monday)", "month": "Month"}[gran]
            st.bar_chart(frame, color=SENT_COLORS, stack=True, height=300, x_label=x_label, y_label="Reviews")
        else:
            st.caption("No data to chart.")

    # ---- table ------------------------------------------------------------
    st.markdown("**Reviews**")
    size_col, page_col, dl_col = st.columns([1, 1, 2])
    page_size = size_col.selectbox("Rows per page", list(config.PAGE_SIZE_OPTIONS[:3]), key="dv_w_size")
    pages = max(1, -(-o["total"] // page_size))
    signature = (tuple(sorted((k, str(v)) for k, v in filters.items())), page_size)
    if ss.get("dv_sig") != signature:
        ss["dv_w_page"] = 1
        ss["dv_sig"] = signature
    ss["dv_w_page"] = min(max(1, int(ss.get("dv_w_page", 1))), pages)
    page = page_col.number_input("Page", min_value=1, max_value=pages, step=1, key="dv_w_page")
    rows, _ = database.fetch_reviews(page=int(page), page_size=page_size, sort="newest", db_path=db, **filters)
    view = _table_frame(rows)
    table_key = f"dv_table_{ss.get('dv_table_n', 0)}"
    event = st.dataframe(
        view,
        hide_index=True,
        width="stretch",
        on_select="rerun",
        selection_mode="multi-row",
        key=table_key,
        column_config={
            "id": st.column_config.NumberColumn("ID", format="%d", width="small"),
            "created_at": st.column_config.TextColumn("Date", width="small"),
            "confidence": st.column_config.NumberColumn("Confidence", format="percent", width="small"),
            "rating": st.column_config.NumberColumn("Rating", format="%.0f", width="small"),
            "text": st.column_config.TextColumn("Review", width="large"),
        },
    )
    _delete_controls(event, view, db)
    dl_col.download_button("Download filtered CSV", _filtered_csv(db, filters, o["total"]), "filtered_reviews.csv", "text/csv")
    st.caption(f"Page {int(page)} of {pages}")


def _delete_controls(event, view: pd.DataFrame, db) -> None:
    """Select rows in the table, then delete them after a confirmation click."""
    ss = st.session_state
    try:
        picked = list(event.selection.rows)
    except (AttributeError, TypeError, KeyError):
        picked = []
    ids = [int(view.iloc[i]["id"]) for i in picked if 0 <= i < len(view)]
    if ids and "dv_pending_delete" not in ss:
        st.caption(f"{len(ids)} selected. IDs: {', '.join(str(i) for i in ids)}")
        if st.button(f"Delete {len(ids)} selected review{'s' if len(ids) != 1 else ''}", icon=":material/delete:", key="dv_delete"):
            ss["dv_pending_delete"] = ids
            st.rerun()
    elif not ids:
        st.caption("Tip: tick the box at the left of a row to select it for deletion.")
    pending = ss.get("dv_pending_delete")
    if pending:
        st.warning(f"Delete {len(pending)} review{'s' if len(pending) != 1 else ''} (ID {', '.join(str(i) for i in pending)}) permanently? This cannot be undone.")
        yes, no, _ = st.columns([1, 1, 3])
        if yes.button("Yes, delete", type="primary", key="dv_delete_yes"):
            for review_id in pending:
                database.delete_review(review_id, db)
            ss.pop("dv_pending_delete", None)
            ss["dv_table_n"] = ss.get("dv_table_n", 0) + 1  # new table key clears the selection
            st.rerun()
        if no.button("Keep them", key="dv_delete_no"):
            ss.pop("dv_pending_delete", None)
            st.rerun()


def _filtered_csv(db, filters: dict, total: int) -> str:
    """Every matching review (up to the import limit) as CSV text."""
    rows: list[dict] = []
    page = 1
    while len(rows) < min(total, config.MAX_CSV_ROWS):
        chunk, _ = database.fetch_reviews(page=page, page_size=500, sort="newest", db_path=db, **filters)
        if not chunk:
            break
        rows.extend(chunk)
        page += 1
    return _table_frame(rows).to_csv(index=False)
