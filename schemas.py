"""Input validation for reviews (manual form and CSV import).

Pure Python (standard library only) so it is easy to unit test. Validation
returns clean records ready for ``database.insert_review(s)`` or a list of
human-readable errors; it never raises for bad user input.
"""
from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

import config

REQUIRED_COLUMNS = ("review_text",)
OPTIONAL_COLUMNS = ("product", "rating", "source", "created_at")
DATE_FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y")

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WHITESPACE = re.compile(r"[ \t]+")


@dataclass
class RowError:
    """A problem found in one CSV row (row_number counts the header as row 1)."""

    row_number: int
    message: str


@dataclass
class CsvResult:
    records: list[dict[str, Any]] = field(default_factory=list)
    errors: list[RowError] = field(default_factory=list)
    total_rows: int = 0
    fatal_error: str | None = None
    ignored_columns: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.fatal_error is None


def clean_text(value: Any) -> str:
    """Strip control characters and surrounding whitespace; keep newlines."""
    if value is None:
        return ""
    text = _CONTROL_CHARS.sub("", str(value))
    text = "\n".join(_WHITESPACE.sub(" ", line).strip() for line in text.splitlines())
    return text.strip()


def parse_rating(value: Any) -> tuple[int | None, str | None]:
    """Return (rating, error). Blank means no rating."""
    text = clean_text(value)
    if text == "":
        return None, None
    try:
        number = float(text)
    except ValueError:
        return None, f"rating '{text}' is not a number"
    if number != int(number):
        return None, f"rating '{text}' must be a whole number"
    rating = int(number)
    if not config.MIN_RATING <= rating <= config.MAX_RATING:
        return None, f"rating {rating} must be between {config.MIN_RATING} and {config.MAX_RATING}"
    return rating, None


def parse_created_at(value: Any, now: datetime | None = None) -> tuple[str | None, str | None]:
    """Return (normalized 'YYYY-MM-DD HH:MM:SS', error). Blank means 'now'."""
    now = now or datetime.now()
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        parsed = datetime(value.year, value.month, value.day)
    else:
        text = clean_text(value)
        if text == "":
            return now.strftime("%Y-%m-%d %H:%M:%S"), None
        parsed = None
        for fmt in DATE_FORMATS:
            try:
                parsed = datetime.strptime(text, fmt)
                break
            except ValueError:
                continue
        if parsed is None:
            return None, f"date '{text}' is not recognised (use YYYY-MM-DD)"
    if parsed > now + timedelta(days=1):
        return None, f"date {parsed:%Y-%m-%d} is in the future"
    return parsed.strftime("%Y-%m-%d %H:%M:%S"), None


_KEYBOARD_ROWS = ("qwertyuiop", "asdfghjkl", "zxcvbnm", "1234567890")
_VOWELS = set("aeiouy")


def _is_gibberish_word(word: str) -> bool:
    """A made-up string such as "mmmmmmmmmmmm", "asdfghjkl" or "hdfjkshd" (real words have vowels and variety)."""
    w = word.lower()
    letters = [c for c in w if c.isalpha()]
    if len(letters) < 5:
        return False
    if len(set(letters)) <= 2:
        return True                                   # "mmmmm", "ababab"
    if re.search(r"(.)\1{4,}", w) and len(set(letters)) <= 3:
        return True
    if not any(c in _VOWELS for c in letters):
        return True                                   # "hdfjksh"
    if len(letters) >= 5 and any(w[i:i + 5] in row or w[i:i + 5] in row[::-1] for row in _KEYBOARD_ROWS for i in range(len(w) - 4)):
        return True                                   # "asdfg", "qwert"
    return False


def check_review_text(value: Any) -> str | None:
    """Why this text is not a real review, or None when it looks like one.

    Deliberately simple and explainable, not a language model: it only catches input that no review could be (a
    keyboard mash, one repeated letter, mostly symbols or digits, a single word). Used for reviews typed into the
    chat or the form; CSV import keeps its own row checks because exported data legitimately has short rows.
    """
    text = clean_text(value)
    if not text:
        return "the review is empty"
    compact = [c for c in text if not c.isspace()]
    letters = [c for c in compact if c.isalpha()]
    if len(letters) < 4 or len(letters) / len(compact) < 0.5:
        return "it is mostly symbols or numbers, or too short to be a review"
    words = re.findall(r"[^\W\d_]+(?:['\u2019-][^\W\d_]+)*", text)
    bad = [w for w in words if _is_gibberish_word(w)]
    if bad and len(bad) * 2 >= len(words):
        return "it looks like random characters, not words"
    if len(words) < 2:
        return "it is a single word; a review needs at least a short phrase"
    return None


def validate_review(
    review_text: Any,
    product: Any = None,
    rating: Any = None,
    source: Any = None,
    created_at: Any = None,
    default_source: str = "manual",
    now: datetime | None = None,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Validate one review. Returns (record, errors); record is None if errors exist."""
    errors: list[str] = []

    text = clean_text(review_text)
    if not text:
        errors.append("review text is empty")
    elif len(text) > config.MAX_REVIEW_LENGTH:
        errors.append(f"review text is {len(text)} characters (maximum {config.MAX_REVIEW_LENGTH})")

    product_clean = clean_text(product) or None
    if product_clean and len(product_clean) > config.MAX_PRODUCT_LENGTH:
        errors.append(f"product name is longer than {config.MAX_PRODUCT_LENGTH} characters")

    rating_value, rating_error = parse_rating(rating)
    if rating_error:
        errors.append(rating_error)

    source_clean = clean_text(source) or default_source
    if len(source_clean) > config.MAX_SOURCE_LENGTH:
        errors.append(f"source is longer than {config.MAX_SOURCE_LENGTH} characters")

    created, date_error = parse_created_at(created_at, now=now)
    if date_error:
        errors.append(date_error)

    if errors:
        return None, errors

    record = {
        "review_text": text,
        "product": product_clean,
        "rating": rating_value,
        "sentiment_label": None,  # filled in by Module 3
        "score_positive": None,
        "score_neutral": None,
        "score_negative": None,
        "topics": None,  # filled in by Module 4
        "source": source_clean,
        "created_at": created,
        "model_version": None,
    }
    return record, []


def validate_csv(file_bytes: bytes, default_source: str = "csv_import", now: datetime | None = None) -> CsvResult:
    """Validate an uploaded CSV. Bad rows are reported; good rows are returned."""
    result = CsvResult()

    if not file_bytes or not file_bytes.strip():
        result.fatal_error = "The file is empty."
        return result
    if len(file_bytes) > config.MAX_CSV_BYTES:
        result.fatal_error = (
            f"The file is {len(file_bytes) / 1024 / 1024:.1f} MB; the limit is "
            f"{config.MAX_CSV_BYTES / 1024 / 1024:.0f} MB."
        )
        return result

    try:
        text = file_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        result.fatal_error = "The file is not valid UTF-8 text. Re-save it as 'CSV UTF-8' and try again."
        return result

    try:
        sample = text[:4096]
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(io.StringIO(text, newline=""), dialect=dialect)
        raw_headers = reader.fieldnames or []
        headers = [clean_text(h).lower() for h in raw_headers]
        if not any(headers):
            result.fatal_error = "No header row found. The first row must contain column names."
            return result

        missing = [c for c in REQUIRED_COLUMNS if c not in headers]
        if missing:
            result.fatal_error = (
                f"Missing required column: {', '.join(missing)}. "
                f"Found columns: {', '.join(h for h in headers if h) or 'none'}."
            )
            return result

        known = set(REQUIRED_COLUMNS) | set(OPTIONAL_COLUMNS)
        result.ignored_columns = [h for h in headers if h and h not in known]
        reader.fieldnames = headers

        for row_number, row in enumerate(reader, start=2):
            result.total_rows += 1
            if result.total_rows > config.MAX_CSV_ROWS:
                result.records.clear()
                result.errors.clear()
                result.fatal_error = f"The file has more than {config.MAX_CSV_ROWS:,} rows. Split it into smaller files."
                return result

            if None in row:
                result.errors.append(RowError(row_number, "row has more fields than the header"))
                continue
            if all(clean_text(v) == "" for v in row.values()):
                result.total_rows -= 1  # silently skip fully blank lines
                continue

            record, errors = validate_review(
                row.get("review_text"),
                row.get("product"),
                row.get("rating"),
                row.get("source"),
                row.get("created_at"),
                default_source=default_source,
                now=now,
            )
            if errors:
                result.errors.append(RowError(row_number, "; ".join(errors)))
            else:
                result.records.append(record)
    except csv.Error as exc:
        result.records.clear()
        result.fatal_error = f"The file could not be parsed as CSV ({exc})."
        return result

    if result.total_rows == 0:
        result.fatal_error = "The file has a header but no data rows."
    return result
