"""Structured logging setup for AgeWell services."""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog
from pythonjsonlogger import jsonlogger


def configure_logging(level: str = "INFO", json: bool = True) -> None:
    """Configure stdlib logging and structlog once for service entrypoints."""
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    if json:
        handler.setFormatter(
            jsonlogger.JsonFormatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    root.addHandler(handler)

    processors: list[Any] = [
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]
    renderer = structlog.processors.JSONRenderer() if json else structlog.dev.ConsoleRenderer()
    processors.append(renderer)

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(root.level),
        cache_logger_on_first_use=True,
    )
