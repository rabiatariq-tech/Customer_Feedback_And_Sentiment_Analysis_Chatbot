from datetime import datetime

import pytest

import database
import schemas
from services import analytics, topics

NOW = datetime(2026, 10, 21, 12, 0, 0)


@pytest.fixture()
def db(tmp_path):
    path = tmp_path / "t.db"
    database.init_db(path)
    return path


def add(db, text, product, rating, day, label, conf=0.9, flags=None):
    rec, errors = schemas.validate_review(text, product=product, rating=rating, created_at=day, now=NOW)
    assert errors == []
    rec.update(sentiment_label=label, score_positive=conf if label == "positive" else 0.05,
               score_neutral=conf if label == "neutral" else 0.05,
               score_negative=conf if label == "negative" else 0.05, flags=flags)
    database.insert_review(rec, db)


def seed(db):
    add(db, "Delivery was late and the box was damaged", "Lamp", 1, "2026-10-01", "negative")
    add(db, "Great lamp, easy to use", "Lamp", 5, "2026-10-02", "positive")
    add(db, "Too expensive for the quality", "Blender", 2, "2026-10-08", "negative", 0.5, "low_confidence")
    add(db, "Support replied quickly and refunded me", "Blender", 4, "2026-10-09", "positive")
    add(db, "It is fine", "Blender", 3, "2026-10-10", "neutral")
    topics.tag_pending(db)


def test_tagging_is_multi_topic_whole_word_and_defaults_to_other():
    assert topics.tag_topics("Delivery was late and the price is too high") == ["delivery", "price"]
    assert topics.tag_topics("Apparently it is fine") == ["other"]  # 'app' must not match inside 'apparently'
    assert topics.tag_topics("The app keeps crashing") == ["usability", "performance"]
    assert topics.tag_topics("") == ["other"]


def test_tag_pending_only_touches_untagged_and_is_repeatable(db):
    seed(db)
    assert topics.tag_pending(db) == 0
    rows, total = database.fetch_reviews(topic="delivery", db_path=db)
    assert total == 1 and rows[0]["topics"] == "delivery,packaging,product quality"
    with pytest.raises(ValueError):
        database.fetch_reviews(topic="a,b", db_path=db)
    assert database.fetch_reviews(topic="deliver", db_path=db)[1] == 0  # exact tag, not substring


def test_overview(db):
    seed(db)
    o = analytics.overview(db)
    assert (o["total"], o["positive"], o["neutral"], o["negative"]) == (5, 2, 1, 2)
    assert o["negative_pct"] == 40.0 and o["avg_rating"] == 3.0 and o["low_confidence"] == 1
    assert analytics.overview(db, product="Lamp")["total"] == 2
    empty = analytics.overview(db, product="Nope")
    assert empty["total"] == 0 and empty["avg_rating"] is None and empty["negative_pct"] == 0.0


def test_trend_product_and_topic_views(db):
    seed(db)
    assert [t["period"] for t in analytics.sentiment_trend("month", db)] == ["2026-10"]
    assert len(analytics.sentiment_trend("day", db)) == 5
    with pytest.raises(ValueError):
        analytics.sentiment_trend("year", db)
    prods = analytics.by_product(db)
    assert prods[0]["product"] in ("Lamp", "Blender") and {p["product"] for p in prods} == {"Lamp", "Blender"}
    tb = {t["topic"]: t for t in analytics.topic_breakdown(db)}
    assert tb["delivery"]["negative"] == 1 and tb["price"]["negative"] == 1
    assert analytics.topic_breakdown(db, sentiment="negative")[0]["negative"] >= 1


def test_compare_periods_and_bad_filters(db):
    seed(db)
    c = analytics.compare_periods("2026-10-08", "2026-10-10", db)
    assert c["previous_range"] == ("2026-10-05", "2026-10-07")
    assert c["current"]["total"] == 3 and c["previous"]["total"] == 0 and c["enough_data"] is False
    with pytest.raises(ValueError):
        analytics.compare_periods("2026-10-10", "2026-10-01", db)
    with pytest.raises(TypeError):
        analytics.overview(db, bogus="x")


def test_examples_respect_filters(db):
    seed(db)
    ex = analytics.examples(2, db, sentiment="negative")
    assert len(ex) == 2 and all(r["sentiment_label"] == "negative" for r in ex)


def test_trend_granularity_and_labels():
    assert analytics.choose_granularity(10) == "day"
    assert analytics.choose_granularity(90) == "week"
    assert analytics.choose_granularity(400) == "month"
    assert analytics.period_label("2026-W41", "week") == "2026-10-12"  # the Monday of that week
    assert analytics.period_label("2026-10-05", "day") == "2026-10-05"
    assert analytics.period_label("2026-10", "month") == "2026-10"
    assert analytics.period_label("garbage", "week") == "garbage"


# ---- "Complaints are falling" must not be claimed from a handful of reviews (reported: 3 of 8 vs 9 of 16) -------------
def test_chance_of_difference_matches_known_values():
    from services import insights
    assert abs(insights.chance_of_difference(3, 8, 9, 16) - 0.67) < 0.01           # her refund numbers: easily luck
    assert insights.chance_of_difference(0, 20, 10, 20) < 0.001                      # a huge gap is not luck
    assert insights.chance_of_difference(5, 10, 5, 10) == 1.0                        # identical shares


def _trend(cn, ct, pn, pt):
    from datetime import date
    from services import insights
    mk = lambda neg, tot: {"negative": neg, "positive": tot - neg, "neutral": 0, "total": tot}
    return insights.describe_trend("returns & refunds reviews", {"recent": mk(cn, ct), "earlier": mk(pn, pt), "days": 28,
                                                                 "cut": date(2026, 10, 1), "last": date(2026, 10, 28)})


def test_a_gap_that_could_be_luck_is_not_called_a_trend():
    t = _trend(3, 8, 9, 16)
    assert "complaints are falling" not in t.lower() and "by chance" in t and "went down" in t and t.startswith("**Not clear yet.**")


def test_a_clear_gap_is_still_called_a_trend():
    assert "complaints are falling" in _trend(5, 40, 30, 60)
    assert "complaints are rising" in _trend(30, 40, 10, 60)
    assert "No clear change" in _trend(10, 40, 26, 100)


def test_pretty_labels_read_like_dates_people_say():
    from services import analytics
    assert analytics.pretty_label("2026-10-05", "day") == "5 Oct"
    assert analytics.pretty_label("2026-W41", "week") == "12 Oct"
    assert analytics.pretty_label("2026-10", "month") == "Oct 2026"
    assert analytics.pretty_label("garbage", "week") == "garbage"
    # the year appears only when the axis crosses a year boundary (months always show it)
    assert analytics.pretty_labels(["2026-10-05", "2026-10-06"], "day") == ["5 Oct", "6 Oct"]
    assert analytics.pretty_labels(["2026-12-30", "2027-01-02"], "day") == ["30 Dec 2026", "2 Jan 2027"]


def test_cap_range_limits_how_many_bars_a_chart_shows():
    from datetime import date
    from services import analytics
    start, capped = analytics.cap_range("day", date(2026, 1, 1), date(2026, 2, 1))
    assert (start, capped) == (date(2026, 1, 1), False)
    start, capped = analytics.cap_range("day", date(2025, 1, 1), date(2026, 1, 31))
    assert capped and (date(2026, 1, 31) - start).days + 1 == 90


def test_sentiment_chart_spec_has_horizontal_labels_top_legend_and_share_mode():
    import sys, types
    sys.modules.setdefault("streamlit", types.ModuleType("streamlit"))
    from ui import charts
    rows = [{"period": "2026-10", "positive": 3, "neutral": 1, "negative": 2}, {"period": "2026-11", "positive": 0, "neutral": 0, "negative": 0}]
    spec = charts.sentiment_chart(rows, "month").to_dict()
    assert spec["encoding"]["x"]["axis"]["labelAngle"] == 0 and spec["encoding"]["color"]["legend"]["orient"] == "top"
    assert spec["encoding"]["y"]["stack"] == "zero"
    assert charts.sentiment_chart(rows, "month", share=True).to_dict()["encoding"]["y"]["stack"] == "normalize"
