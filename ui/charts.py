"""Stacked sentiment bar charts (Altair, bundled with Streamlit): readable dates, a clear legend, optional share view."""
from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from services import analytics
from ui import theme

SENTIMENTS = ("positive", "neutral", "negative")
_COLORS = [theme.SENT[s][0] for s in SENTIMENTS]


def sentiment_chart(rows: list[dict], granularity: str, share: bool = False, height: int = 300) -> alt.Chart:
    """One stacked bar per period. rows need "period" plus positive/neutral/negative counts (zero rows are fine)."""
    periods = [r["period"] for r in rows]
    labels = analytics.pretty_labels(periods, granularity)
    records = []
    for r, label in zip(rows, labels):
        for s in SENTIMENTS:
            records.append({"when": label, "sentiment": s.capitalize(), "reviews": int(r.get(s, 0) or 0), "rank": SENTIMENTS.index(s)})
    frame = pd.DataFrame(records)
    axis_title = {"day": "Day", "week": "Week starting (Monday)", "month": "Month"}[granularity]
    y = alt.Y("reviews:Q", stack="normalize" if share else "zero", title="Share of reviews" if share else "Reviews",
              axis=alt.Axis(format="%", grid=True) if share else alt.Axis(format="d", grid=True, tickMinStep=1))
    return (
        alt.Chart(frame)
        .mark_bar(cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
        .encode(
            x=alt.X("when:O", sort=labels, title=axis_title,
                    axis=alt.Axis(labelAngle=0, labelOverlap="parity", labelPadding=6, ticks=False, domain=False),
                    scale=alt.Scale(paddingInner=0.3, paddingOuter=0.1)),
            y=y,
            color=alt.Color("sentiment:N", title=None, sort=[s.capitalize() for s in SENTIMENTS],
                            scale=alt.Scale(domain=[s.capitalize() for s in SENTIMENTS], range=_COLORS),
                            legend=alt.Legend(orient="top", direction="horizontal", labelFontSize=13, symbolSize=180, padding=4)),
            order=alt.Order("rank:Q"),
            tooltip=[alt.Tooltip("when:O", title=axis_title), alt.Tooltip("sentiment:N", title="Sentiment"),
                     alt.Tooltip("reviews:Q", title="Reviews")],
        )
        .properties(height=height)
        .configure_axis(labelFontSize=12, titleFontSize=12, titleColor=theme.MUTED, labelColor=theme.MUTED, gridColor="#EEF2F7")
        .configure_view(strokeWidth=0)
    )


def show_sentiment_chart(rows: list[dict], granularity: str, share: bool = False, height: int = 300) -> None:
    chart = sentiment_chart(rows, granularity, share, height)
    try:
        st.altair_chart(chart, width="stretch")
    except TypeError:  # older Streamlit
        st.altair_chart(chart, use_container_width=True)
