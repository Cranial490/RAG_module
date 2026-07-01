from __future__ import annotations

import logging
import subprocess
import sys
import textwrap
from pathlib import Path

from fastapi.testclient import TestClient

import main
from memory_module.logging_config import configure_logging

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_response_carries_generated_request_id_header():
    client = TestClient(main.app)
    response = client.get("/strategies/parsers")

    assert response.status_code in (200, 404)
    request_id = response.headers.get("x-request-id")
    assert request_id is not None
    assert request_id != ""


def test_sanitized_inbound_request_id_is_echoed():
    client = TestClient(main.app)
    inbound = "abc-123_def.XYZ"
    response = client.get(
        "/strategies/parsers",
        headers={"X-Request-ID": inbound},
    )

    assert response.headers.get("x-request-id") == inbound


def test_malformed_inbound_request_id_is_replaced():
    client = TestClient(main.app)
    too_long = "a" * 65
    malformed_cases = [
        "has spaces",
        "has/slash",
        "has;semicolon",
        too_long,
    ]

    for malformed in malformed_cases:
        response = client.get(
            "/strategies/parsers",
            headers={"X-Request-ID": malformed},
        )
        returned = response.headers.get("x-request-id")
        assert returned is not None
        assert returned != malformed
        assert 1 <= len(returned) <= 64


def test_sequential_requests_get_distinct_ids():
    client = TestClient(main.app)
    ids = {
        client.get("/strategies/parsers").headers["x-request-id"]
        for _ in range(5)
    }

    assert len(ids) == 5


def test_records_emitted_during_request_carry_request_id(monkeypatch, caplog):
    configure_logging()
    module_logger = logging.getLogger("memory_module.test_correlation")

    def emit_and_return():
        module_logger.info("processing")
        return ["docx"]

    monkeypatch.setattr(main, "list_parsers", emit_and_return)
    main.STRATEGY_LISTERS = {**main.STRATEGY_LISTERS, "parsers": main.list_parsers}

    client = TestClient(main.app)
    caplog.set_level(logging.INFO, logger="memory_module.test_correlation")

    response = client.get("/strategies/parsers")
    outbound_id = response.headers["x-request-id"]

    matching = [r for r in caplog.records if r.name == "memory_module.test_correlation"]
    assert matching, "expected at least one memory_module log record"
    for record in matching:
        assert getattr(record, "request_id", None) == outbound_id


def test_one_access_summary_line_per_request(caplog):
    configure_logging()
    client = TestClient(main.app)
    caplog.set_level(logging.INFO, logger="memory_module.access")

    response = client.get("/strategies/parsers")
    outbound_id = response.headers["x-request-id"]

    access_records = [r for r in caplog.records if r.name == "memory_module.access"]
    assert len(access_records) == 1
    record = access_records[0]
    assert record.levelno == logging.INFO
    assert getattr(record, "request_id", None) == outbound_id
    assert getattr(record, "method", None) == "GET"
    assert getattr(record, "path", None) == "/strategies/parsers"
    assert getattr(record, "status_code", None) == response.status_code
    duration_ms = getattr(record, "duration_ms", None)
    assert isinstance(duration_ms, (int, float))
    assert duration_ms >= 0


def test_log_level_env_drives_root_level(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    configure_logging()
    assert logging.getLogger().getEffectiveLevel() == logging.DEBUG

    monkeypatch.delenv("LOG_LEVEL", raising=False)
    configure_logging()
    assert logging.getLogger().getEffectiveLevel() == logging.INFO


def test_configure_logging_is_idempotent():
    configure_logging()
    handlers_before = len(logging.getLogger().handlers)
    for _ in range(4):
        configure_logging()
    handlers_after = len(logging.getLogger().handlers)
    assert handlers_after == handlers_before


def test_importing_memory_module_does_not_configure_logging():
    script = textwrap.dedent(
        """
        import json
        import logging

        before = len(logging.getLogger().handlers)
        import memory_module  # noqa: F401
        after = len(logging.getLogger().handlers)
        print(json.dumps({"before": before, "after": after}))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    import json as _json
    payload = _json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["after"] == payload["before"]
