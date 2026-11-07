"""Per-product change between two periods, and the chat answer for "which product caused the increase ...?"."""
from datetime import date

import database
from services import chat_parser, chatbot, product_change
from tests.test_chatbot import add, blocks, text_of

TODAY = date(2026, 11, 1)


def seed_months(db):
    def many(n, text, label, product, day):
        for _ in range(n):
            add(db, text, label, product, day)
    many(8, "Great lamp", "positive", "Desk Lamp", "2026-09-10")
    many(6, "Lamp arm sags, poor quality", "negative", "Desk Lamp", "2026-09-12")
    many(5, "Nice blender", "positive", "Blender Pro", "2026-09-11")
    many(5, "Blender jug cracked", "negative", "Blender Pro", "2026-09-15")
    many(2, "Fine chair", "positive", "Wooden Chair", "2026-09-11")
    many(14, "Great lamp", "positive", "Desk Lamp", "2026-10-10")
    many(2, "Lamp arm sags, poor quality", "negative", "Desk Lamp", "2026-10-12")
    many(5, "Nice blender", "positive", "Blender Pro", "2026-10-11")
    many(4, "Blender jug cracked", "negative", "Blender Pro", "2026-10-15")
    many(2, "Fine chair", "positive", "Wooden Chair", "2026-10-11")


def test_previous_window_for_whole_months_partial_months_and_other_periods():
    pw = product_change.previous_window
    assert pw("2026-10-01", "2026-10-31") == ("2026-09-01", "2026-09-30")
    assert pw("2026-03-01", "2026-03-31") == ("2026-02-01", "2026-02-28")
    assert pw("2026-10-01", "2026-10-21") == ("2026-09-01", "2026-09-21")      # a month so far: the same days of last month
    assert pw("2026-11-01", "2026-11-01") == ("2026-10-01", "2026-10-01")
    assert pw("2026-10-15", "2026-10-21") == ("2026-10-08", "2026-10-14")      # anything else: the equally long period before


def test_per_product_changes_add_up_to_the_overall_change(db):
    seed_months(db)
    r = product_change.product_changes(db, ("2026-10-01", "2026-10-31"), ("2026-09-01", "2026-09-30"))
    assert r["overall"]["positive"] == (15, 21) and r["overall"]["negative"] == (11, 6) and r["overall"]["total"] == (26, 27)
    assert sum(x["d_positive"] for x in r["rows"]) == 21 - 15 and sum(x["d_negative"] for x in r["rows"]) == 6 - 11
    chair = next(x for x in r["rows"] if x["product"] == "Wooden Chair")
    assert chair["few"]                                                         # 2 reviews a month: too few to read a change into


def test_wording_says_accounts_for_never_caused_and_skips_tiny_samples(db):
    seed_months(db)
    r = product_change.product_changes(db, ("2026-10-01", "2026-10-31"), ("2026-09-01", "2026-09-30"))
    text = " ".join(product_change.describe(r))
    assert "accounted for by **Desk Lamp** (+6)" in text and "Wooden Chair" not in text
    assert "caused" not in text.lower() and "because" not in text.lower()
    assert len(product_change.describe(r, "negative")) == 1


def test_parser_recognises_which_product_questions():
    p = lambda t: chat_parser.parse(t, ["Desk Lamp", "Blender Pro"], TODAY)
    q = p("In the month of october which product cause the increase in positive reviews and decrease in negative reviews")
    assert q.intent == "compare" and q.group_by == "product" and q.period.date_from == "2026-10-01" and q.sentiment is None
    q = p("which product caused the rise in negative reviews this month")
    assert q.intent == "compare" and q.group_by == "product" and q.sentiment == "negative"
    q = p("what changed for each product since last month")
    assert q.period.label.startswith("this month") and q.previous_period.label.startswith("last month")
    for plain in ("which product has the most complaints", "which product has better reviews", "how many reviews did each product get",
                  "rank products by number of negative reviews"):
        assert p(plain).intent in ("ranking", "count"), plain
    assert p("rank products by number of negative reviews").intent == "ranking"


def test_chat_answer_names_the_products_and_warns_about_cause(db):
    seed_months(db)
    r = chatbot.route("In the month of october which product cause the increase in positive reviews and decrease in negative reviews", db, TODAY)
    t = text_of(r)
    assert "Positive reviews rose from 15 to 21 (+6)" in t and "Negative reviews fell from 11 to 6" in t and "**Desk Lamp**" in t
    assert "not why it changed" in t and "few reviews" in t
    table = blocks(r, "table")[0]
    assert table["columns"] == product_change.TABLE_COLUMNS and table["rows"][0][0] == "Desk Lamp"
    assert r["query"].group_by == "product"


def test_a_period_without_reviews_is_said_plainly(db):
    seed_months(db)
    t = text_of(chatbot.route("which product caused the rise in positive reviews in august", db, TODAY))
    assert "no reviews in either period" in t or "no reviews in" in t
    assert "accounted" not in t


def test_overall_questions_about_change_still_use_the_ordinary_compare(db):
    seed_months(db)
    t = text_of(chatbot.route("compare this month with last month", db, date(2026, 10, 31)))
    assert "Negative reviews" in t and "accounted" not in t
