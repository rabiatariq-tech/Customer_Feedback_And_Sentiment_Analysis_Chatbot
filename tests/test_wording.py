"""The plain-English phrasing helpers: they only turn values into words, so these tests pin the exact wording."""
from datetime import date
from types import SimpleNamespace

import pytest

from services import wording

TODAY = date(2026, 10, 21)


def q(**kw):
    base = dict(sentiment=None, topic=None, product=None, search_terms=[], exclude_topics=[], period=None)
    base.update(kw)
    return SimpleNamespace(**base)


def period(label):
    return SimpleNamespace(label=label)


@pytest.mark.parametrize("label,expected", [
    ("in October (Oct 1 – Oct 31)", "in October"),
    ("in October 2025 (Oct 1 – Oct 31)", "in October 2025"),
    ("last month (Sep 1 – Sep 30)", "last month (Sep 1 – Sep 30)"),
    ("this week so far", "this week so far"),
])
def test_period_phrase(label, expected):
    assert wording.period_phrase(period(label), TODAY) == expected
    assert wording.period_phrase(None, TODAY) == ""


def test_phrase_and_rest():
    query = q(sentiment="negative", topic="price", product="Desk Lamp", period=period("in October (Oct 1 – Oct 31)"))
    assert wording.phrase(query, today=TODAY) == "negative reviews about price for Desk Lamp in October"
    assert wording.phrase(query, sentiment=False, today=TODAY) == "reviews about price for Desk Lamp in October"
    assert wording.rest(query, today=TODAY) == " about price for Desk Lamp in October"
    assert wording.rest(q(), today=TODAY) == ""
    assert wording.phrase(q(exclude_topics=["delivery"], search_terms=["battery"])) == "reviews leaving out delivery mentioning “battery”"


@pytest.mark.parametrize("part,whole,words", [
    (0, 5, "none"), (9, 10, "nearly all"), (3, 4, "most"), (5, 8, "about two thirds"), (19, 33, "a little over half"),
    (14, 28, "about half"), (2, 5, "about four in ten"), (11, 34, "about a third"), (71, 246, "about a quarter"),
    (1, 5, "about a fifth"), (1, 7, "about one in seven"), (1, 10, "about one in ten"), (1, 20, "fewer than one in ten"),
])
def test_proportion_matches_the_number(part, whole, words):
    assert wording.proportion(part, whole) == words


def test_small_helpers():
    assert wording.join_names([]) == "" and wording.join_names(["A"]) == "A"
    assert wording.join_names(["A", "B"]) == "A and B" and wording.join_names(["A", "B", "C"]) == "A, B and C"
    assert wording.plural(1, "review") == "1 review" and wording.plural(1200, "review") == "1,200 reviews"
    assert wording.split_words(42, 10, 17) == "42 positive (61%), 10 neutral (14%) and 17 negative (25%)"
