"""Per-visitor session setup: which database to use, the shared sentiment model and chat state."""
from __future__ import annotations

import uuid
from pathlib import Path

import streamlit as st

import config
import database
from services import feedback_service, import_service, session_db, topics
from services.sentiment import SentimentAnalyzer


@st.cache_resource(show_spinner=False)
def get_analyzer() -> SentimentAnalyzer:
    """One analyzer per server process; the model itself loads once, on first use."""
    return SentimentAnalyzer()


@st.cache_resource(show_spinner=False)
def _llm():
    """(provider, quota guard) shared by the question planner and the summaries, or None without a Gemini key/model.
    One guard means both draw on the same free quota."""
    import config
    from services.llm.provider import GeminiProvider
    from services.llm.quota import QuotaGuard

    key, model = config.GEMINI_API_KEY, config.GEMINI_MODEL
    try:  # Streamlit Cloud keeps secrets here instead of a .env file
        key = key or str(st.secrets.get("GEMINI_API_KEY", "")).strip()
        model = model or str(st.secrets.get("GEMINI_MODEL", "")).strip()
    except Exception:  # noqa: BLE001 - no secrets file locally
        pass
    if not key or not model:
        return None
    guard = QuotaGuard(config.LLM_DAILY_LIMIT, config.LLM_SESSION_LIMIT, minute_limit=config.LLM_MINUTE_LIMIT)
    return GeminiProvider(key, model, config.LLM_TIMEOUT_SECONDS), guard


def get_planner():
    """The AI question planner, or None when no Gemini key/model is configured (rules only)."""
    import config
    from services.planner import QuestionPlanner

    llm = _llm()
    return None if llm is None or config.PLANNER_MODE == "rules" else QuestionPlanner(*llm)


def get_summarizer(session_id: str = "anon"):
    """The AI summary writer for this visitor's session (charged to their session quota), or None when summaries are
    off (AI_SUMMARIES=0) or there is no Gemini key."""
    import functools

    import config
    from services.summarizer import ReviewSummarizer

    llm = _llm()
    return None if llm is None or not config.AI_SUMMARIES else functools.partial(ReviewSummarizer(*llm), session_id=session_id)


def load_model() -> tuple[bool, str | None]:
    """Load the sentiment model, showing a spinner on the first load."""
    analyzer = get_analyzer()
    if analyzer.loaded:
        return True, None
    with st.spinner("Loading the sentiment model... the first run downloads about 500 MB."):
        return analyzer.ensure_loaded()


def db_path() -> Path:
    """The database for this visitor.

    Locally: the normal file (data persists). In DEMO_MODE: a private temporary file per browser session,
    pre-filled with the 30 sample reviews the first time it is created.
    """
    ss = st.session_state
    if "db_path" not in ss:
        ss["session_key"] = uuid.uuid4().hex
        path = session_db.resolve_db_path(ss["session_key"])
        if config.DEMO_MODE and database.count_reviews(path) == 0:
            import_service.load_sample(path)
            ok, _ = load_model()
            if ok:
                feedback_service.analyze_reviews(get_analyzer(), db_path=path)
        topics.tag_pending(path)  # tag anything saved before topics existed
        ss["db_path"] = path
    return ss["db_path"]


def init_chat_state() -> None:
    ss = st.session_state
    ss.setdefault("messages", [])
    ss.setdefault("pending", None)
    ss.setdefault("mode", None)
