"""Integration checks for the Phase 0 backing services."""

import httpx
import psycopg
import pytest

ENDPOINTS = {
    "minio": "http://localhost:9000/minio/health/ready",
    "mlflow": "http://localhost:5000/health",
    "prefect": "http://localhost:4200/api/health",
    "inference": "http://localhost:8000/health",
}


@pytest.mark.integration
@pytest.mark.parametrize(("svc", "url"), ENDPOINTS.items())
def test_http_health(svc: str, url: str) -> None:
    """HTTP services expose healthy status endpoints."""
    response = httpx.get(url, timeout=5.0)
    assert response.status_code in (200, 204), f"{svc} not healthy: {response.text}"


@pytest.mark.integration
def test_postgres_alive(pg_conn) -> None:  # type: ignore[no-untyped-def]
    """Postgres accepts queries."""
    with pg_conn.cursor() as cursor:
        cursor.execute("SELECT 1")
        assert cursor.fetchone() == (1,)


def _assert_pgvector_loaded(conn: psycopg.Connection[tuple], database: str) -> None:
    """Assert that the vector extension is installed in a database."""
    with conn.cursor() as cursor:
        cursor.execute("SELECT extname FROM pg_extension WHERE extname = 'vector'")
        assert cursor.fetchone() is not None, f"pgvector extension missing in {database} DB"


@pytest.mark.integration
def test_pgvector_extension_loaded(pg_conn: psycopg.Connection[tuple]) -> None:
    """Pgvector is installed in the agewell database."""
    _assert_pgvector_loaded(pg_conn, "agewell")


@pytest.mark.integration
def test_agentos_pgvector_extension_loaded(agentos_pg_conn: psycopg.Connection[tuple]) -> None:
    """Pgvector is installed in the agentos database."""
    _assert_pgvector_loaded(agentos_pg_conn, "agentos")
