"""Plain-English phrases for answers: how a filter is said out loud, how a share is put in words, how a date range reads.

Nothing here touches the database or decides anything: it only turns values the code already has into words a
person would use ("negative reviews about price in October", "about a third"), so the answer templates stay short and
every handler talks the same way. Technical descriptions ("understood as: ...") still use ``Query.scope``.
"""
from __future__ import annotations

import re
from datetime import date
from typing import Any

_LABEL_WITH_MONTH = re.compile(r"^in (?P<month>[A-Z][a-z]+)(?: (?P<year>\d{4}))? \(")


def period_phrase(period: Any, today: date | None = None) -> str:
    """"in October" for a named month in this year, "in October 2025" for another year, otherwise the label as it is
    ("last month (Oct 1 – Oct 31)", "this week so far")."""
    if period is None:
        return ""
    label = str(period.label)
    m = _LABEL_WITH_MONTH.match(label)
    if m:
        year = m.group("year")
        this_year = (today or date.today()).year
        return f"in {m.group('month')}" + (f" {year}" if year and int(year) != this_year else "")
    return label


def phrase(q: Any, noun: str = "reviews", *, sentiment: bool = True, exclusions: bool = True, today: date | None = None) -> str:
    """"negative reviews about price for Desk Lamp in October" (no trailing "overall")."""
    head = f"{q.sentiment} {noun}" if (sentiment and q.sentiment) else noun
    parts = [head]
    if exclusions and q.exclude_topics:
        parts.append("leaving out " + " and ".join(q.exclude_topics))
    if q.topic:
        parts.append(f"about {q.topic}")
    if q.product:
        parts.append(f"for {q.product}")
    if q.search_terms:
        parts.append("mentioning " + " or ".join(f"“{t}”" for t in q.search_terms))
    when = period_phrase(q.period, today)
    if when:
        parts.append(when)
    return " ".join(parts)


def rest(q: Any, *, sentiment: bool = False, exclusions: bool = True, today: date | None = None) -> str:
    """Everything after the noun, with a leading space ("" when nothing narrows the reviews down)."""
    full = phrase(q, "@", sentiment=sentiment, exclusions=exclusions, today=today)
    tail = full.split("@", 1)[1] if "@" in full else ""
    return tail


def proportion(part: int, whole: int) -> str:
    """A share in words that matches the number: "about a third" for 30-38%, "about four in ten" for 38-45%, and so on."""
    if whole <= 0 or part <= 0:
        return "none"
    share = part / whole
    for floor, words in ((0.90, "nearly all"), (0.72, "most"), (0.60, "about two thirds"), (0.55, "a little over half"), (0.45, "about half"),
                         (0.38, "about four in ten"), (0.30, "about a third"), (0.22, "about a quarter"), (0.17, "about a fifth"),
                         (0.13, "about one in seven"), (0.09, "about one in ten")):
        if share >= floor:
            return words
    return "fewer than one in ten"


def join_names(names: list[str]) -> str:
    """"A", "A and B", "A, B and C"."""
    names = [n for n in names if n]
    if len(names) <= 1:
        return "".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


def plural(count: int, word: str) -> str:
    return f"{count:,} {word}" + ("" if count == 1 else "s")


def split_words(positive: int, neutral: int, negative: int) -> str:
    """"42 positive (61%), 10 neutral (14%) and 17 negative (25%)"."""
    total = max(positive + neutral + negative, 1)
    return f"{positive} positive ({positive / total:.0%}), {neutral} neutral ({neutral / total:.0%}) and {negative} negative ({negative / total:.0%})"
