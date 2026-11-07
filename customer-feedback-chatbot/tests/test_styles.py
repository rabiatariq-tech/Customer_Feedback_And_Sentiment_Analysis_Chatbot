import pytest

pytest.importorskip("streamlit")

from ui import styles  # noqa: E402


def _row(**overrides):
    row = {
        "id": 7,
        "review_text": "Nice <b>lamp</b>",
        "product": "Lamp",
        "rating": 4,
        "sentiment_label": None,
        "source": "manual",
        "created_at": "2026-10-01 10:30:00",
    }
    row.update(overrides)
    return row


def test_review_card_escapes_html_in_user_text():
    card = styles.review_card(_row(review_text="<script>alert(1)</script>", product="<img src=x>"))
    assert "<script>" not in card
    assert "<img" not in card
    assert "&lt;script&gt;" in card


def test_review_card_shows_pending_badge_and_stars():
    card = styles.review_card(_row())
    assert "Pending analysis" in card
    assert "★★★★☆" in card


def test_review_card_handles_missing_rating_and_product():
    card = styles.review_card(_row(rating=None, product=None))
    assert "No product" in card
    assert "stars" not in card


def test_sentiment_badge_labels():
    assert "Positive" in styles.sentiment_badge("positive")
    assert "Negative" in styles.sentiment_badge("negative")


def test_metric_card_escapes_values():
    assert "<i>" not in styles.metric_card("<i>x</i>", "<b>1</b>")


def test_badge_shows_confidence_only_for_analysed_reviews():
    assert "Positive · 94%" in styles.sentiment_badge("positive", 0.94)
    assert "%" not in styles.sentiment_badge(None, 0.94)


def test_review_card_shows_label_and_confidence_from_scores():
    card = styles.review_card(_row(sentiment_label="negative", score_negative=0.876))
    assert "Negative · 88%" in card


def test_score_bars_clamp_and_format():
    bars = styles.score_bars({"positive": 1.7, "neutral": -0.2})
    assert "width:100.0%" in bars
    assert "0.0%" in bars
    assert bars.count("score-row") == 3


def test_breakdown_bars_handles_empty_counts():
    bars = styles.breakdown_bars({"positive": 0, "neutral": 0, "negative": 0, "unanalyzed": 0})
    assert bars.count("score-row") == 4
    assert "Pending" in bars
