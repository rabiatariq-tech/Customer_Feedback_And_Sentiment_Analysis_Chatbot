"""Saving reviews: confirmed CSV imports, single reviews and the sample file.

Everything is stored first (with topics) so a missing sentiment model can never lose data;
sentiment analysis runs afterwards through ``feedback_service.analyze_reviews``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

import config
import database
import schemas
from services import csv_import, topics


@dataclass
class ImportOutcome:
    imported: int = 0
    duplicates: int = 0
    notes: list[str] = field(default_factory=list)
    errors: list[schemas.RowError] = field(default_factory=list)
    batch_id: str | None = None
    fatal_error: str | None = None


def import_dataframe(
    df: pd.DataFrame,
    mapping: dict[str, Any],
    keep: list[str] | None = None,
    date_order: str | None = None,
    filename: str | None = None,
    db_path: str | Path | None = None,
    now: datetime | None = None,
) -> ImportOutcome:
    """Build, validate and store the rows of a confirmed mapping, then tag topics."""
    built = csv_import.build_records(df, mapping, keep=keep, date_order=date_order, now=now)
    outcome = ImportOutcome(notes=list(built.notes), errors=list(built.errors), batch_id=built.batch_id)
    if not built.records:
        outcome.fatal_error = "None of the rows could be imported."
        return outcome
    inserted, skipped = database.insert_reviews(built.records, db_path)
    outcome.imported, outcome.duplicates = inserted, skipped
    safe_mapping = {k: v for k, v in mapping.items() if k != "allow_sensitive"}
    database.insert_batch(built.batch_id, filename, inserted, safe_mapping, db_path)
    topics.tag_pending(db_path)
    return outcome


def add_single_review(
    text: str, db_path: str | Path | None = None, source: str = "chat", now: datetime | None = None
) -> tuple[int | None, list[str]]:
    """Validate and save one review. Returns (review_id, errors)."""
    record, errors = schemas.validate_review(text, default_source=source, now=now)
    if errors:
        return None, errors
    review_id = database.insert_review(record, db_path)
    topics.tag_pending(db_path)
    return review_id, []


def load_sample(db_path: str | Path | None = None) -> ImportOutcome:
    """Import the bundled 30-review English sample file (extra columns ignored)."""
    outcome = ImportOutcome()
    if not Path(config.SAMPLE_CSV).is_file():
        outcome.fatal_error = "The sample file is missing."
        return outcome
    result = schemas.validate_csv(Path(config.SAMPLE_CSV).read_bytes(), default_source="sample")
    if result.fatal_error:
        outcome.fatal_error = result.fatal_error
        return outcome
    outcome.errors = result.errors
    outcome.imported, outcome.duplicates = database.insert_reviews(result.records, db_path)
    topics.tag_pending(db_path)
    return outcome
