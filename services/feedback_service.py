"""Orchestration between the database and the sentiment analyzer.

Reviews are always saved first and analysed afterwards, so a missing model or an
analysis error can never lose customer feedback. Reviews that could not be
analysed simply stay "pending" and can be retried.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import config
import database
from services.sentiment import SentimentAnalyzer

ProgressCallback = Callable[[int, int], None]


@dataclass
class AnalysisSummary:
    total: int = 0  # reviews selected for analysis
    analyzed: int = 0
    failed: int = 0
    truncated: int = 0  # analysed, but text was longer than the model limit
    errors: list[str] = field(default_factory=list)  # distinct error messages
    fatal_error: str | None = None  # e.g. the model could not be loaded

    @property
    def ok(self) -> bool:
        return self.fatal_error is None and self.failed == 0


def analyze_reviews(
    analyzer: SentimentAnalyzer,
    *,
    reanalyze: bool = False,
    batch_size: int | None = None,
    progress: ProgressCallback | None = None,
    db_path: str | Path | None = None,
) -> AnalysisSummary:
    """Analyse pending reviews (or every review when ``reanalyze`` is True).

    Re-analysis overwrites labels, scores and model_version but keeps the original
    review text, so it is safe to run again after changing the model.
    """
    batch_size = batch_size or config.SENTIMENT_BATCH_SIZE
    summary = AnalysisSummary()

    counts = database.count_by_sentiment(db_path)
    summary.total = sum(counts.values()) if reanalyze else counts["unanalyzed"]
    if summary.total == 0:
        return summary

    ok, error = analyzer.ensure_loaded()
    if not ok:
        summary.fatal_error = error
        return summary

    seen_errors: set[str] = set()
    after_id, done = 0, 0
    while True:
        batch = database.fetch_texts(
            after_id=after_id, limit=batch_size, only_pending=not reanalyze, db_path=db_path
        )
        if not batch:
            break
        results = analyzer.analyze_many([text for _, text in batch], batch_size=batch_size)
        updates = []
        for (review_id, _), result in zip(batch, results):
            if result.ok:
                updates.append((review_id, result.to_record_fields()))
                summary.analyzed += 1
                summary.truncated += int(result.truncated)
            else:
                summary.failed += 1
                if result.error and result.error not in seen_errors:
                    seen_errors.add(result.error)
                    summary.errors.append(result.error)
        if updates:
            database.update_sentiments(updates, db_path)
        after_id = batch[-1][0]  # keyset paging: failed rows are not fetched again this run
        done += len(batch)
        if progress:
            progress(min(done, summary.total), summary.total)
    return summary


def analyze_one(
    analyzer: SentimentAnalyzer, review_id: int, db_path: str | Path | None = None
) -> AnalysisSummary:
    """Analyse a single stored review (used right after the form saves it)."""
    summary = AnalysisSummary(total=1)
    review = database.get_review(review_id, db_path)
    if review is None:
        summary.failed, summary.errors = 1, ["review not found"]
        return summary
    result = analyzer.analyze(review["review_text"])
    if result.ok:
        database.update_sentiment(review_id, result.to_record_fields(), db_path)
        summary.analyzed, summary.truncated = 1, int(result.truncated)
    elif result.error and result.error.startswith("Could not load model"):
        summary.fatal_error = result.error
    else:
        summary.failed, summary.errors = 1, [result.error or "unknown error"]
    return summary


@dataclass
class SavedReview:
    review: dict | None = None  # the stored row (with sentiment and topics when analysis worked)
    errors: list[str] = field(default_factory=list)  # validation problems; nothing was saved
    analysis: AnalysisSummary | None = None


def save_review(
    analyzer: SentimentAnalyzer,
    text: str,
    product: str | None = None,
    rating: int | None = None,
    source: str = "chat",
    db_path: str | Path | None = None,
) -> SavedReview:
    """Validate, save, then analyse and tag one review. The review is saved even if analysis fails."""
    import schemas
    from services import topics as topic_service

    out = SavedReview()
    problem = schemas.check_review_text(text)
    if problem:  # nothing is saved: gibberish would also distort every count and sentiment figure
        out.errors = [f"That does not look like a real review ({problem}). Write a sentence or two about the product, "
                      "for example “The lamp is bright and the cable is long enough.”"]
        return out
    record, errors = schemas.validate_review(text, product=product, rating=rating, source=source)
    if errors:
        out.errors = errors
        return out
    review_id = database.insert_review(record, db_path)
    out.analysis = analyze_one(analyzer, review_id, db_path)
    database.update_topics([(review_id, topic_service.to_storage(topic_service.tag_topics(record["review_text"])))], db_path)
    out.review = database.get_review(review_id, db_path)
    return out
