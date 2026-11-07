"""Streamlit entry point: a chat-first feedback assistant with a separate Data view.

Run:  streamlit run app.py
Storage, validation, sentiment, topics and answers live in database.py, schemas.py and services/;
the files in ui/ only draw the pages.
"""
import streamlit as st

import config
from services import analytics
from ui import chat_page, data_page, nav, session, theme

st.set_page_config(page_title="Feedback Assistant", page_icon="💬", layout="wide", initial_sidebar_state="expanded")

session.init_chat_state()
DB = session.db_path()

nav.CHAT = st.Page(chat_page.render, title="Chat", icon=":material/chat:", url_path="chat", default=True)
nav.DATA = st.Page(data_page.render, title="Data view", icon=":material/table_chart:", url_path="data")
page = st.navigation([nav.CHAT, nav.DATA])

with st.sidebar:
    st.markdown('<div class="brand">Feedback <span>Assistant</span></div>', unsafe_allow_html=True)
    st.caption("Public demo · your data stays in your session" if config.DEMO_MODE else "Local database")
    st.divider()
    o = analytics.overview(DB)
    analysed = max(o["positive"] + o["neutral"] + o["negative"], 1)
    st.markdown(
        f'<div class="side-stat"><span>Reviews</span><b>{o["total"]:,}</b></div>'
        f'<div class="side-stat"><span>Positive</span><b>{o["positive"] / analysed:.0%}</b></div>'
        f'<div class="side-stat"><span>Neutral</span><b>{o["neutral"] / analysed:.0%}</b></div>'
        f'<div class="side-stat"><span>Negative</span><b>{o["negative"] / analysed:.0%}</b></div>',
        unsafe_allow_html=True,
    )
    if o["pending"]:
        st.caption(f"{o['pending']} review(s) not analysed yet.")
    st.divider()
    if st.button("New conversation", icon=":material/add:", width="stretch"):
        st.session_state.messages = []
        st.session_state.pending = None
        st.session_state.mode = None
        st.switch_page(nav.CHAT)
    planner = session.get_planner()
    if planner is None:
        st.caption("Sentiment: local model. Answers: SQL over your data. AI question helper: off (rule-based).")
    else:
        used = planner.guard.status()
        st.caption(f"Sentiment: local model. Answers: SQL over your data. AI question helper: on "
                   f"({used['used_today']} of {used['daily_limit']} calls used today). Only your question is sent, never the reviews.")

page.run()
