"""HTTP clients for imaging services."""

from __future__ import annotations

from typing import Any

import httpx


class ServiceClient:
    """Small JSON-over-HTTP client for service dispatch."""

    def __init__(self, base_url: str, timeout: float = 600.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST JSON and return a decoded response."""
        response = httpx.post(
            f"{self.base_url}/{path.lstrip('/')}",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise TypeError(f"unexpected service response: {type(data).__name__}")
        return data
