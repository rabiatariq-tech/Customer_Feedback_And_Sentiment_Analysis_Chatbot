"""Global CSS and small HTML components so app.py stays focused on behaviour."""
from __future__ import annotations

import html

import streamlit as st

# Palette
INDIGO = "#4F46E5"
VIOLET = "#7C3AED"
TEAL = "#0D9488"
GREEN = "#16A34A"
AMBER = "#D97706"
RED = "#DC2626"
SLATE = "#64748B"

SENTIMENT_COLORS = {
    "positive": (GREEN, "#DCFCE7"),
    "neutral": (SLATE, "#E2E8F0"),
    "negative": (RED, "#FEE2E2"),
    "unanalyzed": (AMBER, "#FEF3C7"),
}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stApp {{ font-family: 'Inter', sans-serif; }}

/* Layout */
.block-container {{ padding-top: 4.2rem !important; padding-bottom: 3rem; max-width: 1200px; }}
header[data-testid="stHeader"] {{ background: rgba(245, 247, 251, .92); backdrop-filter: blur(6px); }}
#MainMenu, footer {{ visibility: hidden; }}
/* Hide the auto-generated anchor-link icon next to headings */
[data-testid="stHeaderActionElements"], h1 a, h2 a, h3 a, .hero a {{ display: none !important; }}

/* Hero banner */
.hero {{
    background: linear-gradient(120deg, {INDIGO} 0%, {VIOLET} 55%, #EC4899 120%);
    border-radius: 20px; padding: 28px 34px; margin-bottom: 22px; color: #fff;
    box-shadow: 0 12px 30px rgba(79, 70, 229, .25);
}}
.hero h1 {{ margin: 0 0 6px 0; font-size: 1.9rem; font-weight: 700; color: #fff; letter-spacing: -.5px; }}
.hero p {{ margin: 0; opacity: .92; font-size: 1rem; }}
.hero .pill {{
    display: inline-block; background: rgba(255,255,255,.18); border: 1px solid rgba(255,255,255,.3);
    padding: 3px 12px; border-radius: 999px; font-size: .78rem; margin-bottom: 12px; font-weight: 500;
}}

/* Metric cards */
.metric-card {{
    position: relative; overflow: hidden; background: #fff; border-radius: 16px; padding: 20px 20px 16px 24px;
    border: 1px solid #E5E9F2; box-shadow: 0 2px 10px rgba(15, 23, 42, .05); height: 100%;
}}
.metric-card::before {{
    content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 6px; background: var(--accent, {INDIGO});
}}
.metric-card .label {{ color: {SLATE}; font-size: .78rem; font-weight: 600; text-transform: uppercase; letter-spacing: .06em; }}
.metric-card .value {{ font-size: 2rem; font-weight: 700; color: #0F172A; line-height: 1.2; margin-top: 4px; }}
.metric-card .hint {{ color: {SLATE}; font-size: .8rem; margin-top: 2px; }}

/* Section headings */
.section-title {{ font-size: 1.15rem; font-weight: 700; color: #0F172A; margin: 6px 0 2px 0; }}
.section-sub {{ color: {SLATE}; font-size: .9rem; margin-bottom: 14px; }}

/* Cards for forms / panels */
div[data-testid="stForm"], .panel {{
    background: #fff; border: 1px solid #E5E9F2; border-radius: 16px; padding: 22px;
    box-shadow: 0 2px 10px rgba(15, 23, 42, .05);
}}

/* Tabs (selectors cover both the older baseweb attributes and the role-based DOM) */
.stTabs [role="tablist"], .stTabs [data-baseweb="tab-list"] {{
    gap: 8px; background: #fff; padding: 8px 10px; border-radius: 14px; margin-bottom: 14px;
    border: 1px solid #E5E9F2 !important; box-shadow: 0 2px 8px rgba(15,23,42,.04);
}}
.stTabs button[role="tab"], .stTabs [data-baseweb="tab"] {{
    border-radius: 10px !important; padding: 10px 22px !important; font-weight: 600; color: {SLATE};
    height: auto !important; min-width: 0; margin: 0 !important; white-space: nowrap;
}}
.stTabs button[role="tab"] p {{ font-weight: 600; font-size: .95rem; margin: 0; }}
.stTabs button[role="tab"]:hover {{ background: #EEF2FF; color: {INDIGO}; }}
.stTabs button[role="tab"][aria-selected="true"], .stTabs [aria-selected="true"] {{
    background: linear-gradient(120deg, {INDIGO}, {VIOLET}) !important; color: #fff !important;
    box-shadow: 0 4px 12px rgba(79,70,229,.28);
}}
.stTabs button[role="tab"][aria-selected="true"] p {{ color: #fff !important; }}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{ display: none !important; }}
.stTabs [role="tablist"]::after, .stTabs [role="tablist"]::before {{ display: none !important; }}

/* Buttons */
.stButton > button, .stFormSubmitButton > button, .stDownloadButton > button {{
    border-radius: 10px; font-weight: 600; border: 1px solid #D5DAE6; transition: all .15s ease;
}}
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primaryFormSubmit"] {{
    background: linear-gradient(120deg, {INDIGO}, {VIOLET}); border: none; color: #fff;
    box-shadow: 0 4px 12px rgba(79,70,229,.3);
}}
.stButton > button:hover, .stFormSubmitButton > button:hover {{ transform: translateY(-1px); }}

/* Inputs */
.stTextInput input, .stTextArea textarea, .stNumberInput input, .stDateInput input,
div[data-baseweb="select"] > div {{ border-radius: 10px !important; }}

/* Sidebar */
section[data-testid="stSidebar"] {{ background: #fff; border-right: 1px solid #E5E9F2; }}

/* Review cards */
.review-card {{
    background: #fff; border: 1px solid #E5E9F2; border-left: 5px solid var(--accent, {SLATE});
    border-radius: 12px; padding: 14px 18px; margin-bottom: 10px; box-shadow: 0 1px 6px rgba(15,23,42,.04);
}}
.review-card .meta {{ color: {SLATE}; font-size: .78rem; margin-bottom: 6px; display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }}
.review-card .text {{ color: #1F2937; font-size: .95rem; line-height: 1.5; white-space: pre-wrap; }}
.badge {{
    display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: .72rem; font-weight: 600;
    background: var(--bg, #E2E8F0); color: var(--fg, #334155);
}}
.stars {{ color: #F59E0B; letter-spacing: 1px; }}

/* Empty state */
.empty {{ text-align: center; padding: 40px 20px; color: {SLATE}; background: #fff; border: 1px dashed #CBD5E1; border-radius: 16px; margin-bottom: 18px; }}

/* Score bars (sentiment confidence-like scores) */
.score-row {{ display: flex; align-items: center; gap: 10px; margin: 6px 0; font-size: .88rem; }}
.score-row .name {{ width: 78px; color: #334155; font-weight: 600; }}
.score-row .track {{ flex: 1; background: #EEF2F7; border-radius: 999px; height: 10px; overflow: hidden; }}
.score-row .fill {{ height: 100%; border-radius: 999px; background: var(--c, {INDIGO}); }}
.score-row .pct {{ width: 48px; text-align: right; color: {SLATE}; font-variant-numeric: tabular-nums; }}
.empty .icon {{ font-size: 2.4rem; }}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str, pill: str = "") -> None:
    pill_html = f'<span class="pill">{html.escape(pill)}</span><br>' if pill else ""
    st.markdown(
        f'<div class="hero">{pill_html}<h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p></div>',
        unsafe_allow_html=True,
    )


def section(title: str, subtitle: str = "") -> None:
    sub = f'<div class="section-sub">{html.escape(subtitle)}</div>' if subtitle else ""
    st.markdown(f'<div class="section-title">{html.escape(title)}</div>{sub}', unsafe_allow_html=True)


def metric_card(label: str, value: str, hint: str = "", accent: str = INDIGO) -> str:
    hint_html = f'<div class="hint">{html.escape(hint)}</div>' if hint else ""
    return (
        f'<div class="metric-card" style="--accent:{accent}">'
        f'<div class="label">{html.escape(label)}</div>'
        f'<div class="value">{html.escape(str(value))}</div>{hint_html}</div>'
    )


def render_metric_cards(cards: list[tuple[str, str, str, str]]) -> None:
    """cards: list of (label, value, hint, accent_color)."""
    cols = st.columns(len(cards))
    for col, (label, value, hint, accent) in zip(cols, cards):
        col.markdown(metric_card(label, value, hint, accent), unsafe_allow_html=True)


def sentiment_badge(label: str | None, confidence: float | None = None) -> str:
    key = label or "unanalyzed"
    fg, bg = SENTIMENT_COLORS.get(key, SENTIMENT_COLORS["unanalyzed"])
    text = "Pending analysis" if key == "unanalyzed" else key.capitalize()
    if confidence is not None and label:
        text += f" · {confidence:.0%}"
    return f'<span class="badge" style="--fg:{fg};--bg:{bg}">{html.escape(text)}</span>'


def score_bars(scores: dict[str, float]) -> str:
    """Horizontal bars for the three class scores (confidence-like, not probabilities of truth)."""
    rows = []
    for name in ("positive", "neutral", "negative"):
        value = max(0.0, min(1.0, float(scores.get(name, 0.0))))
        color = SENTIMENT_COLORS[name][0]
        rows.append(
            f'<div class="score-row"><span class="name">{name.capitalize()}</span>'
            f'<div class="track"><div class="fill" style="width:{value * 100:.1f}%;--c:{color}"></div></div>'
            f'<span class="pct">{value:.1%}</span></div>'
        )
    return "".join(rows)


def breakdown_bars(counts: dict[str, int]) -> str:
    """Count bars for the sentiment breakdown of stored reviews."""
    total = sum(counts.values()) or 1
    rows = []
    for name in ("positive", "neutral", "negative", "unanalyzed"):
        n = counts.get(name, 0)
        color = SENTIMENT_COLORS[name][0]
        label = "Pending" if name == "unanalyzed" else name.capitalize()
        rows.append(
            f'<div class="score-row"><span class="name">{label}</span>'
            f'<div class="track"><div class="fill" style="width:{n / total * 100:.1f}%;--c:{color}"></div></div>'
            f'<span class="pct">{n:,}</span></div>'
        )
    return "".join(rows)


def review_card(row: dict) -> str:
    """HTML for one review. All user-provided values are escaped."""
    key = row.get("sentiment_label") or "unanalyzed"
    accent = SENTIMENT_COLORS.get(key, SENTIMENT_COLORS["unanalyzed"])[0]
    rating = row.get("rating")
    stars = f'<span class="stars">{"★" * rating}{"☆" * (5 - rating)}</span>' if rating else ""
    product = html.escape(row.get("product") or "No product")
    confidence = row.get(f"score_{row['sentiment_label']}") if row.get("sentiment_label") else None
    meta = (
        f'{sentiment_badge(row.get("sentiment_label"), confidence)}'
        f"<span>#{int(row['id'])}</span><span>📦 {product}</span>{stars}"
        f'<span>🗓 {html.escape(str(row["created_at"])[:16])}</span>'
        f'<span>📥 {html.escape(row.get("source") or "")}</span>'
    )
    return (
        f'<div class="review-card" style="--accent:{accent}">'
        f'<div class="meta">{meta}</div>'
        f'<div class="text">{html.escape(row["review_text"])}</div></div>'
    )


def empty_state(icon: str, title: str, message: str) -> None:
    st.markdown(
        f'<div class="empty"><div class="icon">{html.escape(icon)}</div>'
        f"<strong>{html.escape(title)}</strong><br>{html.escape(message)}</div>",
        unsafe_allow_html=True,
    )
