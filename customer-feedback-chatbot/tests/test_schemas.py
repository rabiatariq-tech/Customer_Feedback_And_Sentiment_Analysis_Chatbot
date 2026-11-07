from datetime import datetime

import config
import schemas

NOW = datetime(2026, 10, 21, 12, 0, 0)


def test_valid_review_is_cleaned_and_normalized():
    record, errors = schemas.validate_review(
        "  Great   product!  ", product=" Desk Lamp ", rating="5", created_at="2026-10-01", now=NOW
    )
    assert errors == []
    assert record["review_text"] == "Great product!"
    assert record["product"] == "Desk Lamp"
    assert record["rating"] == 5
    assert record["source"] == "manual"
    assert record["created_at"] == "2026-10-01 00:00:00"
    assert record["sentiment_label"] is None


def test_empty_text_rejected():
    record, errors = schemas.validate_review("   \n ", now=NOW)
    assert record is None
    assert any("empty" in e for e in errors)


def test_text_too_long_rejected():
    record, errors = schemas.validate_review("x" * (config.MAX_REVIEW_LENGTH + 1), now=NOW)
    assert record is None
    assert any("maximum" in e for e in errors)


def test_control_characters_removed():
    record, _ = schemas.validate_review("Good\x00 item\x07", now=NOW)
    assert record["review_text"] == "Good item"


def test_rating_validation():
    assert schemas.parse_rating("") == (None, None)
    assert schemas.parse_rating("4.0") == (4, None)
    assert schemas.parse_rating("6")[1] is not None
    assert schemas.parse_rating("0")[1] is not None
    assert schemas.parse_rating("abc")[1] is not None
    assert schemas.parse_rating("3.5")[1] is not None


def test_date_parsing_and_future_rejection():
    assert schemas.parse_created_at("2026-10-01", now=NOW) == ("2026-10-01 00:00:00", None)
    assert schemas.parse_created_at("", now=NOW)[0] == "2026-10-21 12:00:00"
    assert schemas.parse_created_at("not a date", now=NOW)[1] is not None
    assert "future" in schemas.parse_created_at("2030-01-01", now=NOW)[1]


def test_csv_valid_file():
    data = (
        "review_text,product,rating,created_at\n"
        '"Love it, truly",Lamp,5,2026-10-01\n'
        "Broke quickly,Lamp,1,2026-10-02\n"
    ).encode()
    result = schemas.validate_csv(data, now=NOW)
    assert result.ok
    assert result.total_rows == 2
    assert len(result.records) == 2
    assert result.errors == []
    assert result.records[0]["review_text"] == "Love it, truly"
    assert result.records[0]["source"] == "csv_import"


def test_csv_reports_bad_rows_and_keeps_good_ones():
    data = (
        "review_text,rating\n"
        "Fine,4\n"
        ",3\n"
        "Bad rating,9\n"
        "Another good one,\n"
    ).encode()
    result = schemas.validate_csv(data, now=NOW)
    assert result.ok
    assert len(result.records) == 2
    assert [e.row_number for e in result.errors] == [3, 4]


def test_csv_missing_required_column():
    result = schemas.validate_csv(b"comment,rating\nhello,5\n", now=NOW)
    assert not result.ok
    assert "review_text" in result.fatal_error


def test_csv_empty_and_header_only():
    assert not schemas.validate_csv(b"", now=NOW).ok
    assert "no data rows" in schemas.validate_csv(b"review_text,rating\n", now=NOW).fatal_error


def test_csv_not_utf8():
    result = schemas.validate_csv("review_text\ncafé\n".encode("latin-1"), now=NOW)
    assert not result.ok
    assert "UTF-8" in result.fatal_error


def test_csv_row_limit(monkeypatch):
    monkeypatch.setattr(config, "MAX_CSV_ROWS", 2)
    data = b"review_text\none\ntwo\nthree\n"
    result = schemas.validate_csv(data, now=NOW)
    assert not result.ok
    assert "rows" in result.fatal_error
    assert result.records == []


def test_csv_size_limit(monkeypatch):
    monkeypatch.setattr(config, "MAX_CSV_BYTES", 10)
    result = schemas.validate_csv(b"review_text\nthis is longer than ten bytes\n", now=NOW)
    assert not result.ok


def test_csv_extra_columns_ignored_and_bom_handled():
    data = "﻿review_text,expected_sentiment\nNice,positive\n".encode("utf-8")
    result = schemas.validate_csv(data, now=NOW)
    assert result.ok
    assert result.ignored_columns == ["expected_sentiment"]
    assert len(result.records) == 1


def test_csv_row_with_too_many_fields():
    result = schemas.validate_csv(b"review_text,rating\nok,5\nbad,3,extra\n", now=NOW)
    assert len(result.records) == 1
    assert len(result.errors) == 1


def test_sample_csv_is_valid():
    result = schemas.validate_csv(config.SAMPLE_CSV.read_bytes(), now=NOW)
    assert result.ok
    assert result.errors == []
    assert len(result.records) == 30
