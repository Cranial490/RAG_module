"""Request correlation: request-scoped ID, ASGI middleware, and logging filter.

Library-safe: importing this module does not configure logging handlers.
"""

from __future__ import annotations

import contextvars
import logging
import re
import secrets
import time

_REQUEST_ID_HEADER = b"x-request-id"
_access_logger = logging.getLogger("memory_module.access")
_MAX_ID_LEN = 64
_ALLOWED_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")

_request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "memory_module_request_id", default=None
)


def get_request_id() -> str | None:
    """Return the current request's ID, or None if outside a request."""
    return _request_id_var.get()


def _generate_request_id() -> str:
    return secrets.token_hex(4)


def _sanitize_inbound_id(raw: str | None) -> str | None:
    if raw is None:
        return None
    if not (1 <= len(raw) <= _MAX_ID_LEN):
        return None
    if not _ALLOWED_ID_RE.match(raw):
        return None
    return raw


def _extract_inbound_id(scope) -> str | None:
    for name, value in scope.get("headers", ()):
        if name == _REQUEST_ID_HEADER:
            try:
                return value.decode("latin-1")
            except UnicodeDecodeError:
                return None
    return None


class RequestIDMiddleware:
    """Pure-ASGI middleware that stamps every request with a short request ID.

    The ID is available to every log record via RequestIDFilter and is echoed
    back in the ``X-Request-ID`` response header.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        inbound = _sanitize_inbound_id(_extract_inbound_id(scope))
        request_id = inbound if inbound is not None else _generate_request_id()
        token = _request_id_var.set(request_id)
        header_value = request_id.encode("latin-1")

        status_holder: dict[str, int] = {"code": 0}

        async def send_with_header(message):
            if message["type"] == "http.response.start":
                status_holder["code"] = message.get("status", 0)
                headers = list(message.get("headers", []))
                headers.append((b"x-request-id", header_value))
                message = {**message, "headers": headers}
            await send(message)

        started = time.perf_counter()
        try:
            await self.app(scope, receive, send_with_header)
        finally:
            duration_ms = round((time.perf_counter() - started) * 1000, 3)
            method = scope.get("method", "")
            path = scope.get("path", "")
            status_code = status_holder["code"]
            _access_logger.info(
                "%s %s %s %sms",
                method,
                path,
                status_code,
                duration_ms,
                extra={
                    "method": method,
                    "path": path,
                    "status_code": status_code,
                    "duration_ms": duration_ms,
                },
            )
            _request_id_var.reset(token)


class RequestIDFilter(logging.Filter):
    """Attach the current request ID to every log record as ``request_id``."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id_var.get() or "-"
        return True
