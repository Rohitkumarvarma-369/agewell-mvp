"""Pytest fixtures for AgeWell."""

from collections.abc import Iterator

import psycopg
import pytest


def _connect_postgres(database: str) -> psycopg.Connection[tuple]:
    """Connect to a local Compose Postgres database."""
    return psycopg.connect(
        f"postgresql://agewell:agewell@localhost:5532/{database}",
        connect_timeout=5,
    )


@pytest.fixture
def pg_conn() -> Iterator[psycopg.Connection[tuple]]:
    """Return a Postgres connection to the local Compose agewell database."""
    conn = _connect_postgres("agewell")
    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture
def agentos_pg_conn() -> Iterator[psycopg.Connection[tuple]]:
    """Return a Postgres connection to the local Compose agentos database."""
    conn = _connect_postgres("agentos")
    try:
        yield conn
    finally:
        conn.close()
