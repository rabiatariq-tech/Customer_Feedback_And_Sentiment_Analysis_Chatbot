"""Build data/demo_seed.db: the 246 demo reviews, already analysed by the real sentiment model and topic-tagged.

Why: the public demo gives every visitor a private copy of this file, so the app opens instantly with data in it and
does not have to load the 500 MB model (and its memory) just to show the first screen. The model is only loaded later,
if a visitor adds a review or uploads a CSV.

Run once on your computer (venv active, internet for the first model download), then commit data/demo_seed.db:
    python scripts/build_demo_seed.py
"""
from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
import database  # noqa: E402
import schemas  # noqa: E402
from services import feedback_service, topics  # noqa: E402
from services.sentiment import SentimentAnalyzer  # noqa: E402

SOURCE = config.DATA_DIR / "demo_reviews.csv"
TARGET = config.DEMO_SEED


def main() -> int:
    if not SOURCE.is_file():
        print(f"Missing {SOURCE}")
        return 1
    if TARGET.exists():
        TARGET.unlink()
    database.init_db(TARGET)
    result = schemas.validate_csv(SOURCE.read_bytes(), default_source="demo")
    if result.fatal_error:
        print(result.fatal_error)
        return 1
    inserted, _ = database.insert_reviews(result.records, TARGET)
    print(f"Imported {inserted} reviews. Running the sentiment model (first run downloads it)...")
    summary = feedback_service.analyze_reviews(SentimentAnalyzer(), db_path=TARGET, progress=lambda d, t: print(f"  {d}/{t}", end="\r"))
    if summary.fatal_error:
        print(summary.fatal_error)
        return 1
    topics.tag_pending(TARGET)
    counts = database.count_by_sentiment(TARGET)
    print(f"\nAnalysed {summary.analyzed}, failed {summary.failed}. Sentiment: {dict(counts)}")
    if counts.get("unanalyzed"):
        print("Some reviews were not analysed; fix that before committing the file.")
        return 1
    # Compare with the labels written in the CSV (only a sanity check, the model is not perfect)
    expected = {r["review_text"].strip(): r.get("expected_sentiment", "") for r in csv.DictReader(SOURCE.open(encoding="utf-8"))}
    rows, _ = database.fetch_reviews(page=1, page_size=1000, sort="newest", db_path=TARGET)
    agree = Counter(expected.get(r["review_text"].strip()) == r.get("sentiment_label") for r in rows if expected.get(r["review_text"].strip()))
    if agree:
        print(f"Agreement with the labels in the CSV: {agree[True]}/{agree[True] + agree[False]}")
    import sqlite3
    con = sqlite3.connect(TARGET)
    con.execute("VACUUM")
    con.close()
    print(f"Saved {TARGET} ({TARGET.stat().st_size // 1024} KB). Commit it to GitHub.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
