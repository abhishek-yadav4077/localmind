"""Structured logging setup.

Uses the standard library only. Log lines are ``key=value`` pairs, which are
human-readable in a terminal and still trivially parseable (grep, Loki, etc.).
Pass structured fields with ``logger.info("msg", extra={"model": "llama3.2"})``.
"""

from __future__ import annotations

import logging
import sys

# Attributes every LogRecord has; anything else came from ``extra=`` and is printed.
_RESERVED = set(vars(logging.makeLogRecord({}))) | {"message", "asctime"}


class KeyValueFormatter(logging.Formatter):
    """Format records as ``ts level logger msg key=value ...``."""

    def format(self, record: logging.LogRecord) -> str:
        base = (
            f"{self.formatTime(record, '%Y-%m-%dT%H:%M:%S')} "
            f"level={record.levelname.lower()} logger={record.name} "
            f'msg="{record.getMessage()}"'
        )
        extras = " ".join(
            f"{k}={v!r}" if isinstance(v, str) and " " in v else f"{k}={v}"
            for k, v in record.__dict__.items()
            if k not in _RESERVED
        )
        out = f"{base} {extras}" if extras else base
        if record.exc_info:
            out += "\n" + self.formatException(record.exc_info)
        return out


def configure_logging(level: str = "INFO") -> None:
    """Configure the ``localmind`` logger hierarchy (idempotent)."""
    logger = logging.getLogger("localmind")
    logger.setLevel(level.upper())
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(KeyValueFormatter())
        logger.addHandler(handler)
    # Don't double-log through the root logger (uvicorn/streamlit configure it too).
    logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Return a child of the ``localmind`` logger."""
    return logging.getLogger(name if name.startswith("localmind") else f"localmind.{name}")
