"""Which SQLite file a visitor uses.

Local use: the normal file from config (data persists on your computer).
Public demo (DEMO_MODE=1): a private temporary file per browser session, because free hosts have
temporary disks and visitors must not see each other's uploads. Old temporary files are deleted.
"""
from __future__ import annotations

import re
import tempfile
import time
from pathlib import Path

import config
import database

DEMO_DIR_NAME = "feedback-chatbot-demo"
MAX_AGE_HOURS = 6


def demo_dir(base: Path | None = None) -> Path:
    return (base or Path(tempfile.gettempdir())) / DEMO_DIR_NAME


def purge_stale(base: Path | None = None, max_age_hours: float = MAX_AGE_HOURS, now: float | None = None) -> int:
    """Delete demo databases not touched for ``max_age_hours``. Returns how many files were removed."""
    folder = demo_dir(base)
    if not folder.is_dir():
        return 0
    cutoff = (now if now is not None else time.time()) - max_age_hours * 3600
    removed = 0
    for path in folder.glob("session-*.db*"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
                removed += 1
        except OSError:
            continue
    return removed


def resolve_db_path(session_key: str, demo_mode: bool | None = None, base: Path | None = None) -> Path:
    """Database file for this visitor; creates and migrates it first. Safe to call on every rerun."""
    demo = config.DEMO_MODE if demo_mode is None else demo_mode
    if not demo:
        path = Path(config.DB_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        database.init_db(path)
        return path
    safe = re.sub(r"[^a-zA-Z0-9]", "", session_key)[:32] or "anonymous"
    folder = demo_dir(base)
    folder.mkdir(parents=True, exist_ok=True)
    purge_stale(base)
    path = folder / f"session-{safe}.db"
    database.init_db(path)
    return path
