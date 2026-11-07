"""Chat page: welcome screen, conversation, attachment flow with a mapping card (real services)."""
from __future__ import annotations

import html
from pathlib import Path
import time

import pandas as pd
import streamlit as st

import config
import database
from services import analytics, chatbot, csv_import, feedback_service, import_service, topics
from ui import nav, session, theme

STARTERS = [
    ("Sentiment breakdown", "What is the sentiment breakdown?"),
    ("Top complaints this month", "Summarize the top complaints this month"),
    ("Negative reviews this week", "How many negative reviews this week?"),
    ("Add a review", "I want to add a review"),
]
NONE = "— none —"
_SAMPLE_UPLOAD = (
    "order_ref,feedback_text,customer_email,order_date,stars,branch\n"
    "A-101,Delivery was late and the box arrived damaged.,sara.k@example.com,13/10/2026,2,Lahore\n"
    "A-102,Great sound quality and setup was easy.,bilal@example.com,02/10/2026,9,Karachi\n"
    "A-103,The app keeps crashing and I am very disappointed.,hina_r@example.com,03/10/2026,3,Lahore\n"
    "A-104,Support replied quickly and fixed everything. Impressed!,omar.f@example.com,04/10/2026,10,Islamabad\n"
    "A-105,It works. Nothing special.,ayesha@example.com,05/10/2026,6,Karachi\n"
    "A-106,Too expensive and it stopped working after a week.,zain@example.com,06/10/2026,2,Lahore\n"
    "A-107,,noor@example.com,07/10/2026,5,Islamabad\n"
    "A-108,Fast shipping and perfect packaging. Love it!,mehr@example.com,08/10/2026,9,Karachi\n"
    "A-109,Refund was refused even though the item was defective.,danish@example.com,09/10/2026,1,Lahore\n"
    "A-110,Good value for money and easy to use.,iqra@example.com,10/10/2026,8,Islamabad\n"
).encode("utf-8")
SENT_COLORS = [theme.SENT["positive"][0], theme.SENT["neutral"][0], theme.SENT["negative"][0]]


# ---------------------------------------------------------------------------
# Message helpers
# ---------------------------------------------------------------------------
def _add_user(blocks: list[dict]) -> None:
    st.session_state.messages.append({"role": "user", "blocks": blocks})


def _add_assistant(blocks: list[dict]) -> None:
    st.session_state.messages.append({"role": "assistant", "blocks": blocks, "fresh": True})


def _stream(text: str):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.012)


def _render_block(block: dict, msg_i: int, block_j: int, fresh: bool) -> None:
    kind = block["type"]
    if kind == "text":
        if fresh:
            st.write_stream(_stream(block["text"]))
        else:
            st.markdown(block["text"])
    elif kind == "file":
        st.markdown(theme.file_chip(block["name"]), unsafe_allow_html=True)
    elif kind == "sentiment":
        st.markdown(theme.sentiment_card(block["label"], block["scores"]), unsafe_allow_html=True)
    elif kind == "breakdown":
        st.markdown(theme.bars(block["counts"]), unsafe_allow_html=True)
    elif kind == "chart":
        frame = pd.DataFrame(block["rows"]).set_index("label")[["positive", "neutral", "negative"]]
        st.bar_chart(frame, color=SENT_COLORS, stack=True, height=240, x_label=block.get("x_label", ""), y_label="Reviews")
    elif kind == "table":
        st.dataframe(pd.DataFrame(block["rows"], columns=block["columns"]), hide_index=True, width="stretch")
    elif kind == "reviews":
        st.markdown("".join(theme.review_card(r) for r in block["rows"]), unsafe_allow_html=True)
    elif kind == "evidence":
        with st.expander(f"Evidence · {block['scope']}"):
            st.markdown("".join(theme.review_card(r) for r in block["rows"]), unsafe_allow_html=True)
    elif kind == "badge":
        st.markdown(theme.notice(block["text"], block["kind"]), unsafe_allow_html=True)
    elif kind == "route":
        st.markdown(theme.route_note("Answered with: " + block["text"]), unsafe_allow_html=True)
    elif kind == "actions":
        cols = st.columns(min(len(block["buttons"]), 3) + 1)
        for k, button in enumerate(block["buttons"]):
            if cols[k].button(button["label"], key=f"act_{msg_i}_{block_j}_{k}", icon=":material/table_chart:"):
                nav.open_data(button.get("filters"))


def _render_history() -> None:
    for i, message in enumerate(st.session_state.messages):
        fresh = message.get("fresh", False)
        with st.chat_message(message["role"]):
            for j, block in enumerate(message["blocks"]):
                _render_block(block, i, j, fresh)
        message["fresh"] = False


# ---------------------------------------------------------------------------
# Welcome screen
# ---------------------------------------------------------------------------
def _welcome() -> None:
    st.markdown(
        '<div class="welcome"><h1>How can I help with your customer feedback?</h1>'
        "<p>Attach a CSV with the attachment button in the message box, or ask a question about the "
        "reviews already loaded.</p></div>",
        unsafe_allow_html=True,
    )
    _quick_actions(wide=True)
    with st.expander("Need a file to test with?"):
        st.caption(
            "Sample 1 is a clean file with 30 reviews, 5 products, ratings and dates. Use it to try product questions, "
            "trends and charts."
        )
        try:
            clean_sample = Path(config.SAMPLE_CSV).read_bytes()
        except OSError:
            clean_sample = None
        if clean_sample:
            st.download_button("Download sample 1: clean, with products", clean_sample, "sample_reviews.csv", "text/csv")
        st.caption(
            "Sample 2 has unusual column names, an email column, ambiguous dates and a 1 to 10 rating scale, "
            "and no product column. Use it to see how the mapping step handles messy files."
        )
        st.download_button(
            "Download sample 2: messy columns", _SAMPLE_UPLOAD, "sample_feedback_unusual_columns.csv", "text/csv"
        )


def _quick_actions(wide: bool = False) -> None:
    """The four starter questions. Always available, so they can be used again and again in one conversation."""
    cols = st.columns(2 if wide else len(STARTERS))
    for i, (label, prompt) in enumerate(STARTERS):
        if cols[i % len(cols)].button(label, key=f"quick_{i}", width="stretch"):
            st.session_state.queued = prompt
            st.rerun()


# ---------------------------------------------------------------------------
# Upload and mapping
# ---------------------------------------------------------------------------
def _start_upload(uploaded, user_text: str) -> None:
    ss = st.session_state
    blocks = [{"type": "file", "name": uploaded.name}]
    if user_text:
        blocks.append({"type": "text", "text": user_text})
    _add_user(blocks)

    df, warnings, error = csv_import.read_csv_flexible(uploaded.getvalue())
    if error:
        _add_assistant([{"type": "text", "text": f"I couldn't use that file. {error}"}])
        return
    det = csv_import.detect_columns(df)
    msg = f"I read **{len(df):,} rows** and **{len(df.columns)} columns** from `{uploaded.name}`. "
    if det["text"] and det["confident"]:
        msg += f"`{det['text']}` looks like the review text. Please confirm the mapping below before I import anything."
    elif det["text"]:
        msg += "I'm not sure which column holds the review text, so please choose it below."
    else:
        msg += "I couldn't find a column that looks like review text. Please choose one below, or check the file."
    _add_assistant([{"type": "text", "text": msg}])
    for key in ("map_text", "map_rating", "map_date", "map_product", "map_id", "map_keep", "map_order"):
        ss.pop(key, None)  # a new file must not inherit the previous file's column choices
    ss.pending = {"name": uploaded.name, "df": df, "det": det, "warnings": warnings}


def _mapping_card() -> None:
    ss = st.session_state
    pending = ss.pending
    df, det = pending["df"], pending["det"]
    cols = list(df.columns)
    opts = [NONE] + cols

    def idx(name):
        return opts.index(name) if name in opts else 0

    with st.container(border=True):
        st.markdown("**Confirm how to read this file**")
        for w in pending["warnings"]:
            st.warning(w)
        if det["text"] and det["confident"]:
            st.caption("Suggested automatically. Change anything that looks wrong.")
        elif det["candidates"]:
            guesses = ", ".join(f"`{c}`" for c, _ in det["candidates"])
            st.caption(f"Best guesses for the review text: {guesses}.")

        text_col = st.selectbox(
            "Review text column (required)",
            cols,
            index=cols.index(det["text"]) if det["text"] in cols else None,
            placeholder="Choose the column that holds the review text",
            key="map_text",
        )
        c1, c2, c3, c4 = st.columns(4)
        rating = c1.selectbox("Rating", opts, index=idx(det["rating"]), key="map_rating")
        date_col = c2.selectbox("Date", opts, index=idx(det["date"]), key="map_date")
        product = c3.selectbox("Product", opts, index=idx(det["product"]), key="map_product")
        review_id = c4.selectbox("Review ID", opts, index=idx(det["id"]), key="map_id")

        date_order = None
        if date_col != NONE:
            hint = csv_import.date_format_hint(df[date_col])
            if hint == "ambiguous":
                choice = st.radio(
                    "These dates are ambiguous (for example 03/10/2026). How should I read them?",
                    ["Day / month", "Month / day"],
                    horizontal=True,
                    key="map_order",
                )
                date_order = "dayfirst" if choice == "Day / month" else "monthfirst"
            elif hint in ("dayfirst", "monthfirst"):
                date_order = hint

        chosen = [c for c in (text_col, rating, date_col, product, review_id) if c and c != NONE]
        sensitive = [c for c in det["sensitive"] if c not in chosen]
        extras = [c for c in cols if c not in chosen and c not in sensitive]
        keep = []
        if extras:
            keep = st.multiselect(
                "Other columns to keep (optional)",
                extras,
                default=extras,
                key="map_keep",
                help="Kept columns become filters in the Data view, for example to compare branches or regions. "
                "Click the × on a column to leave it out.",
            )
        if sensitive:
            st.info(
                "Left out for privacy: " + ", ".join(f"`{c}`" for c in sensitive)
                + " (looks like email or phone data). It is never saved or shown in chat."
            )
        st.caption("Preview (first 5 rows)")
        st.dataframe(df.head(5), hide_index=True, width="stretch")

        ok_col, cancel_col, _ = st.columns([1, 1, 3])
        confirm = ok_col.button("Import reviews", type="primary", key="map_ok", width="stretch")
        cancel = cancel_col.button("Cancel", key="map_cancel", width="stretch")

    if cancel:
        _add_assistant([{"type": "text", "text": "No problem, nothing was imported."}])
        ss.pending = None
        st.rerun()
    if confirm:
        if not text_col:
            st.error("Choose the review text column first.")
            return
        if len(chosen) != len(set(chosen)):
            st.error("The same column is used for two different fields. Please pick different columns.")
            return
        mapping = {
            "text": text_col,
            "rating": None if rating == NONE else rating,
            "date": None if date_col == NONE else date_col,
            "product": None if product == NONE else product,
            "id": None if review_id == NONE else review_id,
        }
        _import_confirmed(pending, mapping, keep, date_order)
        ss.pending = None
        st.rerun()


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
def _saved_blocks(saved: feedback_service.SavedReview) -> list[dict]:
    if saved.errors:
        return [{"type": "text", "text": "I couldn't save that review: " + "; ".join(saved.errors)}]
    row = saved.review
    topic_text = ", ".join(topics.from_storage(row.get("topics"))) or "other"
    if row.get("sentiment_label"):
        label = row["sentiment_label"]
        scores = {k: row.get(f"score_{k}") or 0.0 for k in ("positive", "neutral", "negative")}
        blocks = [
            {"type": "text", "text": f"Saved as review #{row['id']}. Predicted sentiment and topic (**{topic_text}**):"},
            {"type": "sentiment", "label": label, "scores": scores},
        ]
        if row.get("flags") and "low_confidence" in row["flags"]:
            blocks.append({"type": "badge", "kind": "rule", "text": "The model is not very sure about this one."})
        return blocks
    reason = (saved.analysis.fatal_error or "; ".join(saved.analysis.errors)) if saved.analysis else "unknown"
    return [{"type": "text", "text": f"Saved as review #{row['id']} (topic: **{topic_text}**), but sentiment could not be "
                                     f"analysed yet ({reason}). It stays pending and can be analysed later."}]


def _save_text_as_review(text: str) -> list[dict]:
    from services import chat_parser

    body, product, rating = chat_parser.extract_inline_review(text) or chat_parser.split_review_fields(text)
    ok, _ = session.load_model()
    with st.spinner("Analysing…"):
        saved = feedback_service.save_review(
            session.get_analyzer(), body, product=product, rating=rating, source="chat", db_path=session.db_path()
        )
    blocks = _saved_blocks(saved)
    if saved.review:
        blocks.append({"type": "text", "text": f"Saved with **ID {saved.review['id']}**. Added by mistake? Say `delete review {saved.review['id']}`, or delete it in the Data view (select the row)."})
    if saved.review and not product:
        blocks.append({"type": "text", "text": "No product was given, so product questions will skip this review. "
                                               "Next time add a line like `Product: Desk Lamp`."})
    return blocks + ([{"type": "actions", "buttons": [{"label": "Open in Data view", "filters": {}}]}] if saved.review else [])


def _handle_text(text: str) -> None:
    ss = st.session_state
    _add_user([{"type": "text", "text": text}])
    from services import chat_parser

    if ss.mode == "delete_confirm":
        ids, ss.mode, ss.delete_ids = ss.get("delete_ids") or [], None, []
        answer = chatbot.parse_confirmation(text)
        if answer == "yes" and ids:
            _add_assistant(chatbot.confirm_delete(ids, session.db_path())["blocks"])
            return
        if answer == "no":
            _add_assistant([{"type": "text", "text": "Okay, nothing was deleted."}])
            return
        # anything else is a new message: handle it normally (the pending delete is dropped)

    if ss.mode == "review_choice":
        choice = text.lower().strip(" .!")
        held, ss.mode = ss.get("review_text") or "", None
        if choice in ("analyze", "analyse", "analysis", "sentiment", "analyze it", "analyse it") and held:
            session.load_model()
            with st.spinner("Analysing…"):
                result = chatbot.route("analyze: " + held, session.db_path(), analyzer=session.get_analyzer())
            _add_assistant(result["blocks"])
            return
        if choice in ("save", "add", "save it", "add it", "yes", "save review") and held:
            _add_assistant(_save_text_as_review(held))
            return
        if choice.startswith(("cancel", "no")):
            _add_assistant([{"type": "text", "text": "Okay, I did nothing with that review."}])
            return
        # anything else is a new message: handle it normally

    if ss.mode == "add_review":
        if text.lower().strip(" .!").startswith("cancel"):
            ss.mode = None
            _add_assistant([{"type": "text", "text": "Okay, I won't save a review."}])
            return
        if not chat_parser.looks_like_command(text):
            ss.mode = None
            _add_assistant(_save_text_as_review(text))
            return
        ss.mode = None  # the message was a question or command, so answer it instead of saving it as a review
    if text.lower().startswith("/review ") and len(text) > 9:
        _add_assistant(_save_text_as_review(text[8:].strip()))
        return
    parsed = chat_parser.parse(text, [], None)
    if parsed.intent == "save_review":
        _add_assistant(_save_text_as_review(text))
        return
    needs_model = parsed.intent == "analyze_text"
    if needs_model:
        session.load_model()
    with st.spinner("Looking at the data…"):
        result = chatbot.route(
            text, session.db_path(), analyzer=session.get_analyzer() if needs_model else None,
            planner=session.get_planner(), session_id=ss.get("session_key", "anon"),
            summarizer=session.get_summarizer(ss.get("session_key", "anon")),
            previous=ss.get("last_query"),
        )
    ss.mode = result["mode"]
    if result.get("query") is not None:
        ss.last_query = result["query"]  # context for a follow-up like "and for September?" (filters only, no review text)
    if result.get("review_text"):
        ss.review_text = result["review_text"]
    if result.get("delete_ids"):
        ss.delete_ids = result["delete_ids"]
    _add_assistant(result["blocks"])


def _import_confirmed(pending: dict, mapping: dict, keep: list[str], date_order: str | None) -> None:
    """Build records from the confirmed mapping, save them, tag topics and analyse sentiment."""
    ss = st.session_state
    db = session.db_path()
    ss.last_query = None  # new data: the previous question's filters no longer apply
    outcome = import_service.import_dataframe(
        pending["df"], mapping, keep=keep, date_order=date_order, filename=pending["name"], db_path=db
    )
    if outcome.fatal_error:
        detail = ""
        if outcome.errors:
            detail = " First problem: row %d: %s." % (outcome.errors[0].row_number, outcome.errors[0].message)
        _add_assistant([{"type": "text", "text": f"{outcome.fatal_error} Nothing was saved.{detail}"}])
        return
    inserted, skipped, built = outcome.imported, outcome.duplicates, outcome

    analysis_note = ""
    ok, load_error = session.load_model()
    if ok:
        bar = st.progress(0.0, text="Analysing sentiment…")
        summary = feedback_service.analyze_reviews(
            session.get_analyzer(), progress=lambda done, total: bar.progress(done / max(total, 1), text=f"Analysing sentiment… {done}/{total}"), db_path=db
        )
        bar.empty()
        if summary.failed or summary.fatal_error:
            analysis_note = f" {summary.failed or 'Some'} review(s) could not be analysed and stay pending."
    else:
        analysis_note = f" The sentiment model could not be loaded, so these reviews stay pending. ({load_error})"

    o = analytics.overview(db, batch_id=built.batch_id)
    text = f"Imported **{inserted:,} reviews** from `{pending['name']}`."
    if o["positive"] + o["neutral"] + o["negative"]:
        text += f" Sentiment: {o['positive']} positive, {o['neutral']} neutral, {o['negative']} negative."
    if o["low_confidence"]:
        text += f" {o['low_confidence']} are low-confidence."
    text += analysis_note
    notes = list(built.notes)
    if skipped:
        notes.append(f"{skipped} duplicate review(s) were already stored and were skipped.")
    if built.errors:
        shown = "; ".join(f"row {e.row_number}: {e.message}" for e in built.errors[:3])
        more = f" (and {len(built.errors) - 3} more)" if len(built.errors) > 3 else ""
        notes.append(f"{len(built.errors)} row(s) were skipped because of problems: {shown}{more}.")
    if notes:
        text += "\n\n" + "\n".join(f"- {n}" for n in notes)
    _add_assistant(
        [{"type": "text", "text": text}, {"type": "actions", "buttons": [{"label": "Open in Data view", "filters": {}}]}]
    )


def render() -> None:
    theme.inject(820)
    ss = st.session_state

    prompt = st.chat_input("Ask about your feedback, or attach a CSV…", accept_file=True, file_type=["csv"])
    text, files = "", []
    if isinstance(prompt, str):
        text = prompt.strip()
    elif prompt:
        text = (prompt.get("text") or "").strip()
        files = list(prompt.get("files") or [])
    queued = ss.pop("queued", None)

    if files:
        _start_upload(files[0], text)
    elif text:
        _handle_text(text)
    elif queued:
        _handle_text(queued)

    empty = not ss.messages and not ss.pending
    if empty:
        _welcome()
    _render_history()
    if ss.pending:
        _mapping_card()
    elif not empty:
        st.caption("Quick questions")
        _quick_actions()
