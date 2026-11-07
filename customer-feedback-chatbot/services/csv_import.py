"""Flexible CSV import: read almost any review CSV, suggest a column mapping, build clean records.

Flow used by the UI:
    1. ``read_csv_flexible(bytes)``      -> DataFrame (all text) + warnings
    2. ``detect_columns(df)``            -> suggested mapping (the user confirms or changes it)
    3. ``build_records(df, mapping, ...)`` -> validated records + notes, ready for ``database.insert_reviews``

Rules: review text is the only required column; extra columns are kept as JSON in ``extra``;
email/phone-like columns are never kept unless the user explicitly asks; nothing here raises on bad files.
"""
from __future__ import annotations

import io
import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

import config
import schemas

_TEXT_NAMES = [
    "review_text", "review", "customer_review", "feedback_text", "feedback", "comment", "comments", "text",
    "description", "remarks", "message", "body", "customer_feedback", "review_body",
]
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_PHONE = re.compile(r"^\+?[\d\s\-()]{9,18}$")
_ID_NAMES = ("id", "review_id", "feedback_id", "reviewid", "order_ref", "order_id", "ref")


def max_rows() -> int:
    return min(config.MAX_CSV_ROWS, config.DEMO_MAX_CSV_ROWS) if config.DEMO_MODE else config.MAX_CSV_ROWS


def new_batch_id() -> str:
    return uuid.uuid4().hex[:12]


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------
def read_csv_flexible(data: bytes) -> tuple[pd.DataFrame | None, list[str], str | None]:
    """Return (dataframe of strings, warnings, error). Never raises for bad files."""
    warnings: list[str] = []
    if not data or not data.strip():
        return None, warnings, "The file is empty."
    if len(data) > config.MAX_CSV_BYTES:
        return None, warnings, (
            f"The file is {len(data) / 1024 / 1024:.1f} MB; the limit is {config.MAX_CSV_BYTES // 1024 // 1024} MB."
        )

    text = None
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            text = data.decode(encoding)
        except UnicodeDecodeError:
            continue
        if encoding != "utf-8-sig":
            warnings.append("The file is not UTF-8, so it was read as Windows-1252. Accented characters may look wrong.")
        break
    if text is None:
        return None, warnings, "The text encoding could not be worked out. Re-save the file as CSV UTF-8 and try again."

    try:
        df = pd.read_csv(io.StringIO(text), sep=None, engine="python", dtype=str)
    except Exception as exc:  # noqa: BLE001 - any parser failure becomes a friendly message
        return None, warnings, f"This does not look like a valid CSV file ({type(exc).__name__}: {exc})."

    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all").reset_index(drop=True)
    if df.empty or len(df.columns) == 0:
        return None, warnings, "The file has no data rows."
    limit = max_rows()
    if len(df) > limit:
        hint = " (public demo limit)" if config.DEMO_MODE else ""
        return None, warnings, f"The file has {len(df):,} rows; the limit is {limit:,}{hint}. Split it into smaller files."
    if len(df.columns) == 1:
        warnings.append("Only one column was found. If the file uses another separator, re-export it as a comma-separated CSV.")
    return df, warnings, None


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
def _numeric_share(series: pd.Series) -> float:
    s = series.dropna()
    return float(pd.to_numeric(s, errors="coerce").notna().mean()) if len(s) else 0.0


def looks_sensitive(name: str, series: pd.Series) -> bool:
    """True for email/phone-like columns (by header or by content)."""
    n = name.lower()
    if any(w in n for w in ("email", "e-mail", "phone", "mobile", "whatsapp", "contact_no")):
        return True
    s = series.dropna().astype(str).str.strip()
    if s.empty:
        return False
    if s.map(lambda v: bool(_EMAIL.match(v))).mean() >= 0.3:
        return True
    phones = s.map(lambda v: bool(_PHONE.match(v)) and sum(ch.isdigit() for ch in v) >= 9)
    return float(phones.mean()) >= 0.5


def _parse_dates(series: pd.Series, order: str | None) -> pd.Series:
    kwargs: dict[str, Any] = {"errors": "coerce"}
    if order == "dayfirst":
        kwargs["dayfirst"] = True
    try:
        return pd.to_datetime(series, format="mixed", **kwargs)
    except (TypeError, ValueError):
        return pd.to_datetime(series, **kwargs)


def date_format_hint(series: pd.Series) -> str:
    """'dayfirst', 'monthfirst' or 'ambiguous' for numeric d/m/y dates, else 'other'."""
    firsts, seconds = [], []
    for value in series.dropna().astype(str):
        m = re.match(r"^\s*(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})\s*$", value)
        if m:
            firsts.append(int(m.group(1)))
            seconds.append(int(m.group(2)))
    if not firsts:
        return "other"
    if any(a > 12 for a in firsts):
        return "dayfirst"
    if any(b > 12 for b in seconds):
        return "monthfirst"
    return "ambiguous"


def detect_columns(df: pd.DataFrame) -> dict[str, Any]:
    """Suggest which column holds the review text, rating, date, product and id.

    ``confident`` is False when the choice is a guess, so the UI should ask the user to confirm.
    """
    scores: list[tuple[str, float]] = []
    sensitive: list[str] = []
    for col in df.columns:
        s = df[col].dropna().astype(str)
        if s.empty:
            continue
        name = col.lower().strip().replace(" ", "_")
        is_sensitive = looks_sensitive(col, df[col])
        if is_sensitive:
            sensitive.append(col)
        score = 0.0
        if name in _TEXT_NAMES:
            score += 3 + (len(_TEXT_NAMES) - _TEXT_NAMES.index(name)) * 0.02
        elif any(w in name for w in ("review", "feedback", "comment", "remark", "opinion", "testimonial")):
            score += 2
        avg_len = float(s.str.len().mean())
        score += min(avg_len, 120) / 120 * 2
        score += (1 - _numeric_share(df[col])) * 0.5
        score += s.nunique() / len(s) * 0.5
        if avg_len < 15:
            score -= 1
        if is_sensitive:
            score -= 3
        scores.append((col, round(score, 2)))
    scores.sort(key=lambda item: -item[1])

    text_col = scores[0][0] if scores and scores[0][1] >= 1.5 else None
    confident = bool(text_col and scores[0][1] >= 2.5 and (len(scores) == 1 or scores[0][1] - scores[1][1] >= 0.8))

    used = {text_col} if text_col else set()

    def pick(words: tuple[str, ...], check) -> str | None:
        for col in df.columns:
            name = col.lower().replace(" ", "_")
            if col in used or col in sensitive:
                continue
            if any(w in name for w in words) and check(df[col]):
                used.add(col)
                return col
        return None

    rating = pick(("rating", "stars", "star", "score", "grade", "rate"), lambda s: _numeric_share(s) >= 0.8)
    date_col = pick(
        ("date", "time", "created", "when", "day"),
        lambda s: float(_parse_dates(s.dropna(), None).notna().mean()) >= 0.8 if len(s.dropna()) else False,
    )
    product = pick(("product", "item", "category", "sku"), lambda s: True)
    review_id = None
    for col in df.columns:
        name = col.lower().replace(" ", "_")
        if col not in used and col not in sensitive and name in _ID_NAMES:
            review_id = col
            used.add(col)
            break
    return {
        "text": text_col,
        "confident": confident,
        "candidates": scores[:3],
        "rating": rating,
        "date": date_col,
        "product": product,
        "id": review_id,
        "sensitive": sensitive,
    }


def normalize_rating(series: pd.Series) -> tuple[pd.Series, str | None]:
    """Scale 1-10 and 1-100 ratings to 1-5. Returns (numbers, note)."""
    nums = pd.to_numeric(series, errors="coerce")
    top = nums.max()
    if pd.isna(top):
        return nums, None
    note = None
    if 5 < top <= 10:
        nums = nums / 2
        note = f"Ratings go up to {top:g}, so they were scaled to 1-5 (divided by 2)."
    elif 10 < top <= 100:
        nums = nums / 20
        note = f"Ratings go up to {top:g}, so they were scaled to 1-5 (divided by 20)."
    elif top > 100:
        return pd.Series(float("nan"), index=series.index), "Ratings go above 100, so the rating column was ignored."
    return nums.round().clip(1, 5), note


# ---------------------------------------------------------------------------
# Building records
# ---------------------------------------------------------------------------
@dataclass
class BuildResult:
    records: list[dict[str, Any]] = field(default_factory=list)
    errors: list[schemas.RowError] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    batch_id: str = ""
    kept_extra: list[str] = field(default_factory=list)
    dropped_extra: list[str] = field(default_factory=list)


def build_records(
    df: pd.DataFrame,
    mapping: dict[str, Any],
    keep: list[str] | None = None,
    date_order: str | None = None,
    batch_id: str | None = None,
    default_source: str = "csv_import",
    now: datetime | None = None,
) -> BuildResult:
    """Turn an uploaded frame and a confirmed mapping into validated records.

    ``mapping`` keys: text (required), rating, date, product. ``keep`` lists extra columns to store
    in ``extra``. Email/phone-like columns are dropped from ``keep`` even if the caller lists them,
    unless the mapping has ``allow_sensitive: True``.
    """
    result = BuildResult(batch_id=batch_id or new_batch_id())
    text_col = mapping.get("text")
    if not text_col or text_col not in df.columns:
        result.notes.append("No review-text column was chosen.")
        return result

    # --- extras -----------------------------------------------------------
    mapped = {mapping.get(k) for k in ("text", "rating", "date", "product")}
    wanted = [c for c in (keep or []) if c in df.columns and c not in mapped]
    allow_sensitive = bool(mapping.get("allow_sensitive"))
    extra_cols: list[str] = []
    for col in wanted:
        if not allow_sensitive and looks_sensitive(col, df[col]):
            result.dropped_extra.append(col)
        else:
            extra_cols.append(col)
    if len(extra_cols) > config.MAX_EXTRA_COLUMNS:
        result.notes.append(f"Only the first {config.MAX_EXTRA_COLUMNS} extra columns were kept.")
        result.dropped_extra.extend(extra_cols[config.MAX_EXTRA_COLUMNS:])
        extra_cols = extra_cols[: config.MAX_EXTRA_COLUMNS]
    # keys must be usable in JSON paths later
    safe_cols = [c for c in extra_cols if c and not any(ch in c for ch in '"\\') and all(ord(ch) >= 32 for ch in c)]
    for c in extra_cols:
        if c not in safe_cols:
            result.dropped_extra.append(c)
    result.kept_extra = safe_cols
    if result.dropped_extra:
        sens = [c for c in result.dropped_extra if c in df.columns and looks_sensitive(c, df[c])]
        if sens:
            result.notes.append("Left out (looks like email/phone data): " + ", ".join(sens) + ".")

    # --- ratings and dates --------------------------------------------------
    rating_series = None
    if mapping.get("rating") in df.columns:
        rating_series, note = normalize_rating(df[mapping["rating"]])
        if note:
            result.notes.append(note)
    parsed_dates = None
    if mapping.get("date") in df.columns:
        parsed_dates = _parse_dates(df[mapping["date"]], date_order)

    # --- rows ---------------------------------------------------------------
    truncated_extra = 0
    empty_text = 0
    bad_dates = 0
    for position, (idx, row) in enumerate(df.iterrows()):
        row_number = position + 2  # header is row 1
        raw_text = row[text_col]
        if pd.isna(raw_text) or not str(raw_text).strip():
            empty_text += 1
            continue

        rating = None
        if rating_series is not None and not pd.isna(rating_series.loc[idx]):
            rating = int(rating_series.loc[idx])

        created = None
        if parsed_dates is not None:
            value = parsed_dates.loc[idx]
            if pd.isna(value):
                if not pd.isna(row[mapping["date"]]) and str(row[mapping["date"]]).strip():
                    bad_dates += 1
            else:
                created = value.to_pydatetime()

        product = row[mapping["product"]] if mapping.get("product") in df.columns else None
        record, errors = schemas.validate_review(
            raw_text,
            None if pd.isna(product) else product,
            rating,
            None,
            created,
            default_source=default_source,
            now=now,
        )
        if errors:
            result.errors.append(schemas.RowError(row_number, "; ".join(errors)))
            continue

        extra: dict[str, str] = {}
        for col in safe_cols:
            value = row[col]
            if pd.isna(value):
                continue
            cleaned = schemas.clean_text(value)
            if not cleaned:
                continue
            if len(cleaned) > config.MAX_EXTRA_VALUE_LENGTH:
                cleaned = cleaned[: config.MAX_EXTRA_VALUE_LENGTH]
                truncated_extra += 1
            extra[col] = cleaned
        record["extra"] = json.dumps(extra, ensure_ascii=False) if extra else None
        record["batch_id"] = result.batch_id
        record["flags"] = None
        result.records.append(record)

    if empty_text:
        result.notes.append(f"Skipped {empty_text} row(s) with empty review text.")
    if bad_dates:
        result.notes.append(f"{bad_dates} date(s) could not be read; those rows use today's date.")
    if truncated_extra:
        result.notes.append(f"{truncated_extra} extra value(s) were shortened to {config.MAX_EXTRA_VALUE_LENGTH} characters.")
    return result
