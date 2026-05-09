"""Pytest fixtures for AgeWell."""

from collections.abc import Iterator

import psycopg
import pytest


@pytest.fixture
def pg_conn() -> Iterator[psycopg.Connection[tuple]]:
    """Return a Postgres connection to the local Compose database."""
    conn = psycopg.connect(
        "postgresql://agewell:agewell@localhost:5532/agewell",
        connect_timeout=5,
    )
    try:
        yield conn
    finally:
        conn.close()
