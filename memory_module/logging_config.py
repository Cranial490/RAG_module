"""Logging setup for entrypoints (API, CLI).

Env-driven. Library modules must NOT call this — they only obtain loggers
via ``logging.getLogger(__name__)``.
"""

from __future__ import annotations

import logging
import os
import sys

from memory_module.logging_correlation import (
    RequestIDFilter,
    get_request_id,
)

_HANDLER_MARKER = "_memory_module_stdout_handler"
_factory_installed = False


def _install_request_id_record_factory() -> None:
    global _factory_installed
    if _factory_installed:
        return

    previous = logging.getLogRecordFactory()

    def factory(*args, **kwargs):
        record = previous(*args, **kwargs)
        record.request_id = get_request_id() or "-"
        return record

    logging.setLogRecordFactory(factory)
    _factory_installed = True


def _remove_marked_handlers(root: logging.Logger) -> None:
    for handler in list(root.handlers):
        if getattr(handler, _HANDLER_MARKER, False):
            root.removeHandler(handler)


def configure_logging(level: str | None = None, log_format: str | None = None) -> None:
    """Configure process-wide logging. Idempotent; safe to call multiple times.

    Reads ``LOG_LEVEL`` (default ``INFO``) and ``LOG_FORMAT`` (default ``text``).
    Currently only ``text`` output is emitted; the ``LOG_FORMAT`` seam is
    reserved for a later JSON slice.
    """
    resolved_level = (level or os.getenv("LOG_LEVEL") or "INFO").upper()
    resolved_format = (log_format or os.getenv("LOG_FORMAT") or "text").lower()
    if resolved_format != "text":
        raise ValueError(
            f"LOG_FORMAT={resolved_format!r} is not yet supported; only 'text' is available."
        )

    _install_request_id_record_factory()

    root = logging.getLogger()
    root.setLevel(resolved_level)

    _remove_marked_handlers(root)
    handler = logging.StreamHandler(sys.stdout)
    setattr(handler, _HANDLER_MARKER, True)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s [%(request_id)s] %(message)s"
        )
    )
    handler.addFilter(RequestIDFilter())
    root.addHandler(handler)

    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
