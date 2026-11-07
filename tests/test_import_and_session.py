import os
import time
from datetime import date, datetime

import pytest

import config
import database
import schemas
from services import chatbot, import_service, session_db, topics
from services.feedback_service import analyze_reviews
from services.sentiment import SentimentAnalyzer
from tests.fakes import FakePipeline

TODAY = date(2026, 10, 21)
NOW = datetime(2026, 10, 21, 12, 0, 0)


@pytest.fixture()
def db(tmp_path):
    path = tmp_path / "t.db"
    database.init_db(path)
    return path


def add(db, text, product, day, label, conf=0.9):
    rec, errors = schemas.validate_review(text, product=product, created_at=day, now=NOW)
    assert errors == []
    rec.update(sentiment_label=label, score_positive=conf if label == "positive" else 0.05,
               score_neutral=conf if label == "neutral" else 0.05,
               score_negative=conf if label == "negative" else 0.05)
    database.insert_review(rec, db)


def seed(db):
    add(db, "Delivery was late and the box was damaged", "Lamp", "2026-10-20", "negative")
    add(db, "Great lamp, easy to use", "Lamp", "2026-10-19", "positive")
    add(db, "Too expensive for the quality", "Blender", "2026-10-18", "negative", 0.7)
    add(db, "Support replied quickly and refunded me", "Blender", "2026-10-02", "positive")
    add(db, "It is fine", "Blender", "2026-09-10", "neutral")
    topics.tag_pending(db)


def kinds(result):
    return [b["type"] for b in result["blocks"]]


def text_of(result):
    return " ".join(b.get("text", "") for b in result["blocks"] if b["type"] == "text")


# ----- import service -----------------------------------------------------------
def test_add_single_review_and_validation(db):
    review_id, errors = import_service.add_single_review("The app keeps crashing", db, now=NOW)
    assert errors == [] and review_id
    row = database.get_review(review_id, db)
    assert row["source"] == "chat" and "usability" in row["topics"]
    assert import_service.add_single_review("   ", db)[0] is None


def test_load_sample_then_analyze_pending(db):
    outcome = import_service.load_sample(db)
    assert outcome.imported == 30 and outcome.fatal_error is None
    assert import_service.load_sample(db).imported == 0  # duplicates skipped
    summary = analyze_reviews(SentimentAnalyzer(pipeline=FakePipeline()), db_path=db)
    assert summary.analyzed == 30
    r = chatbot.route("what is the sentiment breakdown", db, today=TODAY)
    assert sum([b for b in r["blocks"] if b["type"] == "breakdown"][0]["counts"].values()) == 30


def test_import_dataframe_keeps_extras_and_records_batch(db):
    from services import csv_import

    data = (
        "feedback_text,customer_email,stars,branch\n"
        "Delivery was late,a@example.com,2,Lahore\n"
        "Great sound,b@example.com,5,Karachi\n"
    ).encode()
    df, _, error = csv_import.read_csv_flexible(data)
    assert error is None
    out = import_service.import_dataframe(
        df, {"text": "feedback_text", "rating": "stars"}, keep=["branch", "customer_email"], filename="x.csv", db_path=db, now=NOW
    )
    assert out.imported == 2 and out.batch_id
    assert database.list_batches(db)[0]["filename"] == "x.csv"
    rows, _ = database.fetch_reviews(extra_key="branch", extra_value="Lahore", db_path=db)
    assert len(rows) == 1 and "a@example.com" not in (rows[0]["extra"] or "")
    assert rows[0]["topics"]


def test_import_without_valid_rows_reports_error(db):
    import pandas as pd

    out = import_service.import_dataframe(pd.DataFrame({"t": ["", None]}), {"text": "t"}, db_path=db)
    assert out.fatal_error and out.imported == 0


# ----- per-visitor databases ------------------------------------------------------
@pytest.fixture(autouse=True)
def _no_real_seed(tmp_path, monkeypatch):
    """The real data/demo_seed.db (once built) must not leak into tests that expect an empty visitor database."""
    monkeypatch.setattr(config, "DEMO_SEED", tmp_path / "no-seed.db")


def test_new_demo_visitor_starts_from_a_copy_of_the_seed(tmp_path):
    seed = tmp_path / "seed.db"
    database.init_db(seed)
    database.insert_review(schemas.validate_review("Seeded review", now=NOW)[0], seed)
    a = session_db.resolve_db_path("a", demo_mode=True, base=tmp_path / "x", seed=seed)
    b = session_db.resolve_db_path("b", demo_mode=True, base=tmp_path / "x", seed=seed)
    assert database.count_reviews(a) == 1 and database.count_reviews(b) == 1
    database.insert_review(schemas.validate_review("Only for A", now=NOW)[0], a)
    assert database.count_reviews(a) == 2 and database.count_reviews(b) == 1 and database.count_reviews(seed) == 1
    # coming back later must not overwrite the visitor's own changes
    assert session_db.resolve_db_path("a", demo_mode=True, base=tmp_path / "x", seed=seed) == a and database.count_reviews(a) == 2


def test_missing_seed_gives_an_empty_database(tmp_path):
    p = session_db.resolve_db_path("a", demo_mode=True, base=tmp_path, seed=tmp_path / "missing.db")
    assert database.count_reviews(p) == 0


def test_demo_mode_gives_each_visitor_a_separate_database(tmp_path):
    a = session_db.resolve_db_path("visitor-a", demo_mode=True, base=tmp_path)
    b = session_db.resolve_db_path("visitor-b", demo_mode=True, base=tmp_path)
    assert a != b and a.parent == b.parent
    database.insert_review(schemas.validate_review("Only for A", now=NOW)[0], a)
    assert database.count_reviews(a) == 1 and database.count_reviews(b) == 0
    assert session_db.resolve_db_path("visitor-a", demo_mode=True, base=tmp_path) == a


def test_session_key_cannot_escape_the_folder(tmp_path):
    p = session_db.resolve_db_path("../../etc/passwd", demo_mode=True, base=tmp_path)
    assert p.parent == session_db.demo_dir(tmp_path)


def test_stale_demo_databases_are_deleted(tmp_path):
    old = session_db.resolve_db_path("old", demo_mode=True, base=tmp_path)
    fresh = session_db.resolve_db_path("fresh", demo_mode=True, base=tmp_path)
    long_ago = time.time() - 10 * 3600
    os.utime(old, (long_ago, long_ago))
    assert session_db.purge_stale(tmp_path) == 1
    assert not old.exists() and fresh.exists()


def test_local_mode_uses_the_configured_file(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "data" / "mine.db")
    path = session_db.resolve_db_path("anything", demo_mode=False)
    assert path == tmp_path / "data" / "mine.db" and path.exists()
