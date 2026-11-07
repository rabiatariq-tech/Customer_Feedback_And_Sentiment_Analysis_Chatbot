"""Shared test fixtures. A test file can still define its own `db` fixture, which then takes priority."""
import pytest

import database


@pytest.fixture()
def db(tmp_path):
    """A fresh, empty SQLite database file for one test."""
    path = tmp_path / "test.db"
    database.init_db(path)
    return path
