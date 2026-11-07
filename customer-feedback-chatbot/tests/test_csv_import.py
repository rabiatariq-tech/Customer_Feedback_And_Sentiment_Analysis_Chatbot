import json
import sqlite3
from datetime import datetime

import pytest

import config
import database
from services import csv_import

NOW = datetime(2026, 10, 21, 12, 0, 0)

MESSY = (
    "order_ref,feedback_text,customer_email,order_date,stars,branch\n"
    "A-1,Delivery was late and the box arrived damaged.,sara@example.com,13/10/2026,2,Lahore\n"
    "A-2,Great sound quality and easy setup.,bilal@example.com,02/10/2026,9,Karachi\n"
    "A-3,,noor@example.com,03/10/2026,5,Lahore\n"
    "A-4,It works. Nothing special.,ayesha@example.com,05/10/2026,6,Karachi\n"
)


def load(data=MESSY):
    df, warnings, error = csv_import.read_csv_flexible(data.encode("utf-8") if isinstance(data, str) else data)
    assert error is None
    return df, warnings


def test_detects_unusual_headers_and_flags_email():
    df, _ = load()
    m = csv_import.detect_columns(df)
    assert m["text"] == "feedback_text"
    assert m["rating"] == "stars"
    assert m["date"] == "order_date"
    assert "customer_email" in m["sensitive"]
    assert m["id"] == "order_ref"


def test_semicolon_and_cp1252_files_are_read():
    raw = "review;rating\nCafé was lovely;5\nToo slow;2\n".encode("cp1252")
    df, warnings = load(raw)
    assert list(df.columns) == ["review", "rating"]
    assert len(df) == 2
    assert any("Windows-1252" in w for w in warnings)


def test_bad_files_return_errors_not_exceptions():
    assert csv_import.read_csv_flexible(b"")[2] == "The file is empty."
    big = b"a,b\n" + b"1,2\n" * 10
    assert csv_import.read_csv_flexible(big)[2] is None


def test_row_limit_and_demo_mode(monkeypatch):
    monkeypatch.setattr(config, "DEMO_MODE", True)
    monkeypatch.setattr(config, "DEMO_MAX_CSV_ROWS", 3)
    data = "review\n" + "\n".join(f"text {i}" for i in range(5)) + "\n"
    df, _, error = csv_import.read_csv_flexible(data.encode())
    assert df is None and "limit is 3" in error


def test_date_hint():
    df, _ = load()
    assert csv_import.date_format_hint(df["order_date"]) == "dayfirst"
    import pandas as pd

    assert csv_import.date_format_hint(pd.Series(["01/02/2026", "03/04/2026"])) == "ambiguous"
    assert csv_import.date_format_hint(pd.Series(["2026-01-02"])) == "other"


def test_rating_scale_is_normalised():
    import pandas as pd

    nums, note = csv_import.normalize_rating(pd.Series(["2", "9", "10"]))
    assert list(nums) == [1, 4, 5] and "divided by 2" in note
    nums, note = csv_import.normalize_rating(pd.Series(["40", "100"]))
    assert list(nums) == [2, 5]
    nums, note = csv_import.normalize_rating(pd.Series(["1", "500"]))
    assert nums.isna().all() and note


def test_build_records_keeps_extras_and_drops_email():
    df, _ = load()
    m = csv_import.detect_columns(df)
    mapping = {"text": m["text"], "rating": m["rating"], "date": m["date"], "product": None}
    res = csv_import.build_records(
        df, mapping, keep=["branch", "customer_email", "order_ref"], date_order="dayfirst", batch_id="b1", now=NOW
    )
    assert len(res.records) == 3
    assert res.kept_extra == ["branch", "order_ref"]
    assert "customer_email" in res.dropped_extra
    assert any("email" in n for n in res.notes)
    assert any("empty review text" in n for n in res.notes)
    first = res.records[0]
    assert first["created_at"].startswith("2026-10-13")
    assert json.loads(first["extra"]) == {"branch": "Lahore", "order_ref": "A-1"}
    assert first["batch_id"] == "b1"
    assert all("sara@example.com" not in (r["extra"] or "") for r in res.records)
    assert [r["rating"] for r in res.records] == [1, 4, 3]


def test_email_can_be_kept_only_when_explicitly_allowed():
    df, _ = load()
    mapping = {"text": "feedback_text", "allow_sensitive": True}
    res = csv_import.build_records(df, mapping, keep=["customer_email"], now=NOW)
    assert "customer_email" in res.kept_extra


def test_long_extras_and_unsafe_column_names():
    import pandas as pd

    df = pd.DataFrame({"review": ["fine"], 'we"ird': ["x"], "note": ["y" * 900]})
    res = csv_import.build_records(df, {"text": "review"}, keep=['we"ird', "note"], now=NOW)
    assert res.kept_extra == ["note"]
    assert len(json.loads(res.records[0]["extra"])["note"]) == config.MAX_EXTRA_VALUE_LENGTH
    assert any("shortened" in n for n in res.notes)


def test_missing_text_column_gives_note():
    df, _ = load()
    res = csv_import.build_records(df, {"text": None})
    assert res.records == [] and res.notes


# ----- database side ---------------------------------------------------------
@pytest.fixture()
def db(tmp_path):
    path = tmp_path / "t.db"
    database.init_db(path)
    return path


def import_messy(db):
    df, _ = load()
    mapping = {"text": "feedback_text", "rating": "stars", "date": "order_date"}
    res = csv_import.build_records(df, mapping, keep=["branch"], date_order="dayfirst", batch_id="b1", now=NOW)
    database.insert_reviews(res.records, db)
    database.insert_batch("b1", "messy.csv", len(res.records), mapping, db)
    return res


def test_extra_filters_and_helpers(db):
    import_messy(db)
    rows, total = database.fetch_reviews(extra_key="branch", extra_value="Lahore", db_path=db)
    assert total == 1 and rows[0]["review_text"].startswith("Delivery")
    assert database.get_extra_keys(db) == [{"key": "branch", "rows": 3, "distinct_values": 2}]
    assert dict(database.get_extra_values("branch", db_path=db)) == {"Karachi": 2, "Lahore": 1}
    _, total = database.fetch_reviews(batch_id="b1", db_path=db)
    assert total == 3
    assert database.list_batches(db)[0]["filename"] == "messy.csv"


def test_extra_key_cannot_inject_sql(db):
    import_messy(db)
    for bad in ['x") OR 1=1 --', 'a"b', "back\\slash", "nul\x00"]:
        with pytest.raises(ValueError):
            database.fetch_reviews(extra_key=bad, extra_value="x", db_path=db)
    # a value that looks like SQL is just data
    _, total = database.fetch_reviews(extra_key="branch", extra_value="' OR 1=1 --", db_path=db)
    assert total == 0


def test_flag_filter_validates_and_matches(db):
    import_messy(db)
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE feedback SET flags='low_confidence,truncated' WHERE id=1")
        conn.execute("UPDATE feedback SET flags='truncated' WHERE id=2")
    assert database.fetch_reviews(flag="low_confidence", db_path=db)[1] == 1
    assert database.fetch_reviews(flag="truncated", db_path=db)[1] == 2
    with pytest.raises(ValueError):
        database.fetch_reviews(flag="x,%", db_path=db)


def test_migration_preserves_old_database(tmp_path):
    path = tmp_path / "old.db"
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE feedback (
              id INTEGER PRIMARY KEY AUTOINCREMENT, review_text TEXT NOT NULL, product TEXT, rating INTEGER,
              sentiment_label TEXT, score_positive REAL, score_neutral REAL, score_negative REAL, topics TEXT,
              source TEXT NOT NULL DEFAULT 'manual', created_at TEXT NOT NULL, model_version TEXT
            );
            INSERT INTO feedback (review_text, product, rating, sentiment_label, source, created_at)
            VALUES ('Old review', 'Lamp', 4, 'positive', 'manual', '2026-09-01 10:00:00');
            """
        )
    database.init_db(path)
    database.init_db(path)  # twice: must be idempotent
    rows, total = database.fetch_reviews(db_path=path)
    assert total == 1 and rows[0]["review_text"] == "Old review" and rows[0]["sentiment_label"] == "positive"
    assert rows[0]["extra"] is None and rows[0]["flags"] is None
    assert database.get_extra_keys(path) == []
