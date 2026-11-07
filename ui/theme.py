"""Colour tokens, CSS and small HTML components for the chat UI (approved design)."""
from __future__ import annotations

import html

import streamlit as st

PRIMARY = "#0F766E"
PRIMARY_DARK = "#115E59"
PRIMARY_TINT = "#E6F4F1"
BG = "#FFFFFF"
SURFACE = "#F8FAFC"
BORDER = "#E2E8F0"
TEXT = "#0F172A"
MUTED = "#64748B"

# label -> (text colour, tint background)
SENT = {
    "positive": ("#15803D", "#E8F5EC"),
    "neutral": ("#475569", "#EEF1F5"),
    "negative": ("#B91C1C", "#FDECEC"),
}

_CSS = """
<style>
/* Streamlit's fixed header is 3.75rem tall; the page padding must be larger or the first row is hidden under it */
.block-container { max-width: __WIDTH__px; padding-top: 5rem; padding-bottom: 6rem; }
#MainMenu, footer, [data-testid="stDecoration"] { display: none; }
html, body, .stApp, button, input, textarea {
    font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
}
h1, h2, h3 { letter-spacing: -0.01em; color: __TEXT__; }
[data-testid="stHeaderActionElements"] { display: none; }

/* Sidebar */
section[data-testid="stSidebar"] { background: __SURFACE__; border-right: 1px solid __BORDER__; }
.brand { font-weight: 700; font-size: 1.05rem; color: __TEXT__; }
.brand span { color: __PRIMARY__; }
.side-stat { display: flex; justify-content: space-between; font-size: .88rem; color: __MUTED__; padding: 2px 0; }
.side-stat b { color: __TEXT__; font-weight: 600; }

/* Chat messages: assistant plain, user in a soft bubble */
div[data-testid="stChatMessage"] { background: transparent; padding: .5rem 0; gap: .75rem; }
div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: __TINT__; border-radius: 16px; padding: .65rem 1rem; margin-left: 12%;
}
div[data-testid="stChatMessage"] p { line-height: 1.6; }

/* Composer */
div[data-testid="stChatInput"] {
    border: 1px solid __BORDER__; border-radius: 18px; box-shadow: 0 2px 14px rgba(15, 23, 42, .07);
    background: __BG__;
}
div[data-testid="stChatInput"]:focus-within { border-color: __PRIMARY__; }

/* Buttons */
.stButton > button, .stDownloadButton > button {
    border-radius: 12px; border: 1px solid __BORDER__; background: __BG__; color: __TEXT__; font-weight: 500;
    transition: border-color .15s, background .15s;
}
.stButton > button:hover, .stDownloadButton > button:hover { border-color: __PRIMARY__; color: __PRIMARY_DARK__; background: __TINT__; }
.stButton > button[kind="primary"] { background: __PRIMARY__; border-color: __PRIMARY__; color: #fff; }
.stButton > button[kind="primary"]:hover { background: __PRIMARY_DARK__; color: #fff; }

/* Welcome */
.welcome { text-align: center; padding: 3.2rem 0 1.4rem 0; }
.welcome h1 { font-size: 1.9rem; font-weight: 650; margin: 0 0 .5rem 0; }
.welcome p { color: __MUTED__; font-size: 1rem; margin: 0 auto; max-width: 520px; line-height: 1.55; }

/* Cards */
.card { background: __BG__; border: 1px solid __BORDER__; border-radius: 14px; padding: 14px 16px; margin: 6px 0; }
.metric { background: __BG__; border: 1px solid __BORDER__; border-radius: 14px; padding: 14px 16px; }
.metric .label { color: __MUTED__; font-size: .78rem; font-weight: 600; text-transform: uppercase; letter-spacing: .05em; }
.metric .value { font-size: 1.7rem; font-weight: 650; color: __TEXT__; line-height: 1.25; }
.metric .hint { color: __MUTED__; font-size: .8rem; }
.badge { display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: .75rem; font-weight: 600; background: var(--bg); color: var(--fg); }
.chip { display: inline-flex; gap: 8px; align-items: center; background: __SURFACE__; border: 1px solid __BORDER__; border-radius: 10px; padding: 4px 10px; font-size: .85rem; color: __TEXT__; }
.route { color: __MUTED__; font-size: .78rem; margin-top: 6px; }
.route::before { content: ""; display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: __PRIMARY__; margin-right: 6px; }
.notice { border-left: 3px solid var(--c); background: var(--bg); padding: 8px 12px; border-radius: 8px; font-size: .86rem; color: __TEXT__; margin-bottom: 6px; }
.bar-row { display: flex; align-items: center; gap: 10px; margin: 7px 0; font-size: .88rem; }
.bar-row .name { width: 76px; color: __TEXT__; font-weight: 600; }
.bar-row .track { flex: 1; height: 10px; background: #EEF2F7; border-radius: 999px; overflow: hidden; }
.bar-row .fill { height: 100%; border-radius: 999px; background: var(--c); }
.bar-row .val { width: 92px; text-align: right; color: __MUTED__; font-variant-numeric: tabular-nums; }
.review { border: 1px solid __BORDER__; border-left: 4px solid var(--c); border-radius: 10px; padding: 10px 14px; margin: 6px 0; background: __BG__; }
.review .meta { color: __MUTED__; font-size: .76rem; margin-bottom: 4px; display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }
.review .body { font-size: .92rem; line-height: 1.5; color: __TEXT__; }
.page-title { font-size: 1.6rem; font-weight: 650; margin: 0; color: __TEXT__; }
.page-sub { color: __MUTED__; font-size: .92rem; margin-bottom: .8rem; }
</style>
"""


def inject(width: int) -> None:
    css = _CSS.replace("__WIDTH__", str(width))
    for token, value in {
        "__PRIMARY_DARK__": PRIMARY_DARK, "__PRIMARY__": PRIMARY, "__TINT__": PRIMARY_TINT, "__SURFACE__": SURFACE,
        "__BORDER__": BORDER, "__TEXT__": TEXT, "__MUTED__": MUTED, "__BG__": BG,
    }.items():
        css = css.replace(token, value)
    st.markdown(css, unsafe_allow_html=True)


def sentiment_badge(label: str | None, confidence: float | None = None) -> str:
    key = label if label in SENT else "neutral"
    fg, bg = SENT[key]
    text = key.capitalize() + (f" · {confidence:.0%}" if confidence is not None and confidence == confidence else "")
    return f'<span class="badge" style="--fg:{fg};--bg:{bg}">{html.escape(text)}</span>'


def metric_card(label: str, value: str, hint: str = "") -> str:
    hint_html = f'<div class="hint">{html.escape(hint)}</div>' if hint else ""
    return (
        f'<div class="metric"><div class="label">{html.escape(label)}</div>'
        f'<div class="value">{html.escape(str(value))}</div>{hint_html}</div>'
    )


def bars(counts: dict[str, int]) -> str:
    total = sum(counts.values()) or 1
    rows = []
    for name in ("positive", "neutral", "negative"):
        n = counts.get(name, 0)
        color = SENT[name][0]
        rows.append(
            f'<div class="bar-row"><span class="name">{name.capitalize()}</span>'
            f'<div class="track"><div class="fill" style="width:{n / total * 100:.1f}%;--c:{color}"></div></div>'
            f'<span class="val">{n:,} · {n / total:.0%}</span></div>'
        )
    return f'<div class="card">{"".join(rows)}</div>'


def score_bars(scores: dict[str, float]) -> str:
    rows = []
    for name in ("positive", "neutral", "negative"):
        v = max(0.0, min(1.0, float(scores.get(name, 0))))
        rows.append(
            f'<div class="bar-row"><span class="name">{name.capitalize()}</span>'
            f'<div class="track"><div class="fill" style="width:{v * 100:.1f}%;--c:{SENT[name][0]}"></div></div>'
            f'<span class="val">{v:.1%}</span></div>'
        )
    return "".join(rows)


def sentiment_card(label: str, scores: dict[str, float], extra: str = "") -> str:
    return (
        f'<div class="card">{sentiment_badge(label, scores.get(label))} {html.escape(extra)}'
        f'<div style="height:8px"></div>{score_bars(scores)}'
        f'<div class="route" style="margin-top:4px">Scores are model confidence-like values, not guaranteed probabilities.</div></div>'
    )


def review_card(row: dict) -> str:
    label = row.get("sentiment") if row.get("sentiment") in SENT else "neutral"
    meta = (
        f'{sentiment_badge(label, row.get("confidence"))}'
        f'<span>#{int(row["id"])}</span>'
        + (f'<span>{html.escape(str(row["product"]))}</span>' if row.get("product") else "")
        + f'<span>{html.escape(str(row.get("date", "")))}</span>'
    )
    return (
        f'<div class="review" style="--c:{SENT[label][0]}"><div class="meta">{meta}</div>'
        f'<div class="body">{html.escape(str(row["text"]))}</div></div>'
    )


def file_chip(name: str) -> str:
    return f'<span class="chip">📎 {html.escape(name)}</span>'


def route_note(text: str) -> str:
    return f'<div class="route">{html.escape(text)}</div>'


def notice(text: str, kind: str = "ai") -> str:
    color, bg = (PRIMARY, PRIMARY_TINT) if kind == "ai" else ("#B45309", "#FEF3C7")
    return f'<div class="notice" style="--c:{color};--bg:{bg}">{html.escape(text)}</div>'
