"""Shared page handles so any page can switch to another and pre-apply Data view filters."""
from __future__ import annotations

import streamlit as st

CHAT = None  # set by app.py on every run
DATA = None


def open_data(filters: dict | None = None) -> None:
    """Open the Data view with filters from a chat answer (e.g. {'sentiment': 'negative', 'days': 7})."""
    ss = st.session_state
    ss["dv_pref"] = dict(filters or {})
    for key in [k for k in list(ss.keys()) if str(k).startswith("dv_w_")]:
        del ss[key]  # drop old widget state so the new filters take effect
    st.switch_page(DATA)
