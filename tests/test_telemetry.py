"""Telemetry integration tests for AxorMiddleware."""
from __future__ import annotations

import asyncio
import sys

import pytest


@pytest.fixture(autouse=True)
def _isolate_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("AXOR_NO_BANNER", raising=False)
    monkeypatch.delenv("AXOR_TELEMETRY", raising=False)
    monkeypatch.setenv("AXOR_TELEMETRY_QUEUE", str(tmp_path / "queue.jsonl"))

    from axor_langchain import middleware as mw_mod
    monkeypatch.setattr(mw_mod, "_TELEMETRY_MARKER", tmp_path / ".axor" / ".notice")

    try:
        from axor_telemetry import config as tcfg
        monkeypatch.setattr(tcfg, "_CONFIG_PATH", tmp_path / ".axor" / "config.toml")
    except ImportError:
        pass

    return tmp_path


def test_default_ctor_shows_stderr_notice_once(capsys):
    from axor_langchain import AxorMiddleware
    AxorMiddleware()
    captured = capsys.readouterr()
    assert "telemetry is off" in captured.err
    # Second construction — marker suppresses the notice.
    AxorMiddleware()
    captured2 = capsys.readouterr()
    assert "telemetry is off" not in captured2.err


def test_no_banner_env_suppresses_notice(monkeypatch, capsys):
    monkeypatch.setenv("AXOR_NO_BANNER", "1")
    from axor_langchain import AxorMiddleware
    AxorMiddleware()
    captured = capsys.readouterr()
    assert captured.err == ""


def test_env_enables_telemetry_pipeline(monkeypatch):
    monkeypatch.setenv("AXOR_TELEMETRY", "local")
    from axor_langchain import AxorMiddleware
    mw = AxorMiddleware()
    assert mw._telemetry_mode == "local"
    assert mw._telemetry is not None
    assert mw._telemetry.enabled is True


def test_kwarg_wins_over_env(monkeypatch):
    monkeypatch.setenv("AXOR_TELEMETRY", "remote")
    from axor_langchain import AxorMiddleware
    mw = AxorMiddleware(telemetry="local")
    assert mw._telemetry_mode == "local"


def test_off_mode_builds_no_pipeline():
    from axor_langchain import AxorMiddleware
    mw = AxorMiddleware()
    assert mw._telemetry_mode == "off"
    assert mw._telemetry is None


def test_record_telemetry_writes_to_local_queue(monkeypatch, _isolate_home):
    """End-to-end: LC middleware → telemetry pipeline → FileTelemetrySink."""
    monkeypatch.setenv("AXOR_TELEMETRY", "local")
    from langchain_core.messages import AIMessage, HumanMessage
    from axor_langchain import AxorMiddleware

    mw = AxorMiddleware()
    state = {"messages": [
        HumanMessage(content="write a test for the payment endpoint"),
        AIMessage(content="Sure, here's a test..."),
    ]}

    asyncio.run(mw._record_telemetry(state))

    queue_path = _isolate_home / "queue.jsonl"
    assert queue_path.is_file()
    import json
    lines = queue_path.read_text().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["classifier_used"] == "heuristic"
    assert rec["signal_chosen"].startswith("focused")
    # fingerprint was attached
    assert rec["fingerprint_kind"] == "minhash_v1"
    assert rec["fingerprint"] is not None
    assert len(rec["fingerprint"]) == 128


def test_record_telemetry_noop_when_off(_isolate_home):
    """With telemetry off, no queue file is created."""
    from axor_langchain import AxorMiddleware
    mw = AxorMiddleware()
    state = {"messages": []}
    asyncio.run(mw._record_telemetry(state))
    assert not (_isolate_home / "queue.jsonl").exists()
