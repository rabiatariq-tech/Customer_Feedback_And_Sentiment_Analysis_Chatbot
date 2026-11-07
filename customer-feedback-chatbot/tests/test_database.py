from datetime import datetime

import pytest

import database
import schemas

NOW = datetime(2026, 10, 21, 12, 0, 0)


@pytest.fixture()
def db(tmp_path):
    path = tmp_path / "test.db"
    database.init_db(path)
    return path


def make(text, product="Lamp", rating=5, created_at="2026-10-01", source="manual"):
    record, errors = schemas.validate_review(
        text, product=product, rating=rating, created_at=created_at, source=source, now=NOW
    )
    assert errors == []
    return record


def seed(db):
    records = [
        make("Great lamp", "Lamp", 5, "2026-10-01"),
        make("Broke fast", "Lamp", 1, "2026-10-05"),
        make("Loud blender", "Blender", 3, "2026-10-10"),
        make("100% satisfied_customer", "Blender", 5, "2026-10-15"),
    ]
    database.insert_reviews(records, db)


def test_init_is_idempotent_and_empty(db):
    database.init_db(db)
    assert database.count_reviews(db) == 0
    assert database.get_date_range(db) == (None, None)


def test_insert_and_get_review(db):
    review_id = database.insert_review(make("Works well"), db)
    row = database.get_review(review_id, db)
    assert row["review_text"] == "Works well"
    assert row["rating"] == 5
    assert row["sentiment_label"] is None
    assert database.get_review(9999, db) is None


def test_data_persists_across_connections(db):
    database.insert_review(make("Persistent"), db)
    assert database.count_reviews(db) == 1  # new connection each call


def test_bulk_insert_skips_duplicates(db):
    records = [make("Same"), make("Same"), make("Different")]
    inserted, skipped = database.insert_reviews(records, db)
    assert (inserted, skipped) == (2, 1)
    inserted, skipped = database.insert_reviews([make("Same")], db)
    assert (inserted, skipped) == (0, 1)
    assert database.count_reviews(db) == 2


def test_bulk_insert_can_allow_duplicates(db):
    inserted, skipped = database.insert_reviews([make("Same"), make("Same")], db, skip_duplicates=False)
    assert (inserted, skipped) == (2, 0)


def test_filter_by_product_and_date(db):
    seed(db)
    rows, total = database.fetch_reviews(product="Lamp", db_path=db)
    assert total == 2
    rows, total = database.fetch_reviews(date_from="2026-10-05", date_to="2026-10-10", db_path=db)
    assert total == 2
    assert {r["review_text"] for r in rows} == {"Broke fast", "Loud blender"}


def test_filter_by_sentiment_and_unanalyzed(db):
    seed(db)
    assert database.fetch_reviews(sentiment="unanalyzed", db_path=db)[1] == 4
    assert database.fetch_reviews(sentiment="negative", db_path=db)[1] == 0


def test_search_is_case_insensitive_and_escapes_wildcards(db):
    seed(db)
    assert database.fetch_reviews(search="LAMP", db_path=db)[1] == 2  # review text "Great lamp" and the Lamp product rows
    assert database.fetch_reviews(search="100%", db_path=db)[1] == 1
    assert database.fetch_reviews(search="%", db_path=db)[1] == 1  # literal percent, not a wildcard
    assert database.fetch_reviews(search="satisfied_c", db_path=db)[1] == 1


def test_sql_injection_text_is_treated_as_data(db):
    seed(db)
    rows, total = database.fetch_reviews(search="'; DROP TABLE feedback; --", db_path=db)
    assert total == 0
    assert database.count_reviews(db) == 4
    rows, total = database.fetch_reviews(product="Lamp' OR '1'='1", db_path=db)
    assert total == 0


def test_pagination_and_sorting(db):
    seed(db)
    page1, total = database.fetch_reviews(page=1, page_size=3, db_path=db)
    page2, _ = database.fetch_reviews(page=2, page_size=3, db_path=db)
    assert total == 4
    assert len(page1) == 3 and len(page2) == 1
    assert page1[0]["review_text"] == "100% satisfied_customer"  # newest first
    oldest, _ = database.fetch_reviews(sort="oldest", page_size=1, db_path=db)
    assert oldest[0]["review_text"] == "Great lamp"
    low, _ = database.fetch_reviews(sort="rating_low", page_size=1, db_path=db)
    assert low[0]["rating"] == 1
    # unknown sort keys fall back safely instead of reaching SQL
    fallback, _ = database.fetch_reviews(sort="id; DROP TABLE feedback", page_size=1, db_path=db)
    assert len(fallback) == 1


def test_products_and_date_range(db):
    seed(db)
    assert database.get_products(db) == ["Blender", "Lamp"]
    assert database.get_date_range(db) == ("2026-10-01", "2026-10-15")


def test_table_rejects_invalid_values(db):
    import sqlite3

    bad = make("ok")
    bad["rating"] = 9
    with pytest.raises(sqlite3.IntegrityError):
        database.insert_review(bad, db)
    assert database.count_reviews(db) == 0
