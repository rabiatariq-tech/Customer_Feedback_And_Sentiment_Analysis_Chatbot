"""Everyday ideas that reviewers put in different words ("damaged" = crushed, cracked, broke ...).

A review search for "damaged" should also find "arrived crushed", so a searched word is expanded to related word stems before the
SQL LIKE search. Plain lists, no model. Every stem is matched as part of a word (so "crush" finds "crushed" and "crushing").
"""
from __future__ import annotations

import re

CONCEPTS: dict[str, tuple[str, ...]] = {
    r"damag\w*|broken|broke|defect\w*|faulty": ("damag", "crush", "broke", "broken", "crack", "scratch", "dent", "torn", "ripped", "defect", "faulty", "shatter", "smash"),
    r"slow\w*|late|delay\w*": ("slow", "late", "delay", "lag", "sluggish", "took days", "took weeks"),
    r"rude\w*|impolite|unhelpful": ("rude", "impolite", "unhelpful", "dismissive", "disrespect", "arrogant", "unprofessional"),
    r"expensive|overpriced|pricey|costly": ("expensive", "overpriced", "pricey", "costly", "too much", "rip-off", "ripoff", "not worth"),
    r"cheap\w*|flimsy": ("cheap", "flimsy", "fragile", "poorly made", "low quality"),
    r"batter\w*|charg\w*": ("batter", "charge", "charging"),
    r"leak\w*": ("leak", "spill"),
    r"noisy|noise|loud": ("noise", "noisy", "loud", "buzz", "static", "crackl"),
    r"refund\w*": ("refund", "money back", "reimburs"),
}
_GENERIC_TAIL = {"life", "issue", "issues", "problem", "problems", "thing", "things", "item", "items", "stuff", "complaint", "complaints", "review", "reviews"}
_COMPILED = [(re.compile(pat, re.I), stems) for pat, stems in CONCEPTS.items()]


def normalize(term: str) -> str:
    """"battery life" -> "battery", "damaged items" -> "damaged": drop filler words that no review is likely to contain."""
    words = [w for w in term.lower().split() if w]
    while len(words) > 1 and words[-1] in _GENERIC_TAIL:
        words.pop()
    return " ".join(words)


def expand(terms: list[str], limit: int = 12) -> list[str]:
    """The searched words plus related stems, without duplicates, at most ``limit`` (the database allows that many)."""
    out: list[str] = []
    for raw in terms:
        term = normalize(raw)
        if term in _GENERIC_TAIL:  # "life" on its own (from "battery life") is not something a review says
            continue
        for candidate in [term] + [s for rx, stems in _COMPILED if rx.fullmatch(term) for s in stems]:
            if candidate and candidate not in out:
                out.append(candidate)
    return out[:limit]
