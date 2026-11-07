"""Rule-based topic tagging for English reviews (no model, no network, instant).

A review can have several topics ("Delivery was late and the price is too high" -> delivery, price).
Reviews that match nothing get "other", so "topics IS NULL" always means "not tagged yet".
Matching uses whole words/phrases, so "app" does not match "apparently".
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import database

TOPIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "delivery": (
        "deliver", "delivered", "delivery", "shipping", "shipped", "shipment", "courier", "package",
        "parcel", "arrived", "arrive", "late", "tracking", "dispatch",
    ),
    "packaging": (
        "packaging", "packed", "packing", "wrapped", "wrapping", "box", "boxes", "sealed", "unboxing",
    ),
    "product quality": (
        "quality", "broke", "broken", "defect", "defective", "durable", "sturdy", "flimsy", "cheap", "cheaply",
        "stopped working", "cracked", "damaged", "faulty", "fell apart", "build",
    ),
    "price": (
        "price", "priced", "expensive", "overpriced", "cost", "costs", "value", "money", "affordable", "cheaper",
        "worth", "discount", "pricey",
    ),
    "usability": (
        "app", "setup", "set up", "instructions", "easy to use", "confusing", "pair", "pairing", "interface",
        "intuitive", "manual", "user friendly", "hard to use", "difficult to use",
    ),
    "customer support": (
        "support", "service", "agent", "helpdesk", "replied", "reply", "warranty",
        "replacement", "customer care", "complaint",
    ),
    "returns & refunds": (
        "refund", "refunds", "refunded", "return", "returns", "returned", "returning", "exchange", "exchanged", "money back",
    ),
    "performance": (
        "battery", "slow", "fast", "speed", "lag", "laggy", "crash", "crashes", "crashing", "freezes", "sound",
        "noise", "noisy", "loud", "performance", "connection", "disconnects", "bluetooth", "wifi",
    ),
}
OTHER = "other"
TOPICS = tuple(TOPIC_KEYWORDS) + (OTHER,)

_PATTERNS = {
    topic: re.compile(r"\b(?:" + "|".join(re.escape(w) for w in words) + r")\b", re.IGNORECASE)
    for topic, words in TOPIC_KEYWORDS.items()
}


def tag_topics(text: str) -> list[str]:
    """Topics found in one review, in the fixed order of TOPIC_KEYWORDS; ['other'] if none match."""
    found = [topic for topic, pattern in _PATTERNS.items() if pattern.search(text or "")]
    return found or [OTHER]


def to_storage(topics: Sequence[str]) -> str:
    return ",".join(topics)


def from_storage(value: str | None) -> list[str]:
    return [t for t in (value or "").split(",") if t]


def tag_pending(db_path: str | Path | None = None, batch_size: int = 200) -> int:
    """Tag every review that has no topics yet. Returns how many reviews were tagged."""
    tagged = 0
    after_id = 0
    while True:
        batch = database.fetch_untagged(after_id=after_id, limit=batch_size, db_path=db_path)
        if not batch:
            return tagged
        database.update_topics([(i, to_storage(tag_topics(t))) for i, t in batch], db_path)
        tagged += len(batch)
        after_id = batch[-1][0]
