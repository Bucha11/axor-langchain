"""Unit tests for stand-alone helpers in axor_langchain.middleware."""
from __future__ import annotations

import pytest


def test_msg_text_extracts_string_content():
    from axor_langchain.middleware import _msg_text
    class Msg: content = "hello"
    assert _msg_text(Msg()) == "hello"


def test_msg_text_extracts_block_list_content():
    from axor_langchain.middleware import _msg_text
    class Msg:
        content = [
            {"type": "text", "text": "part one "},
            {"type": "text", "text": "part two"},
            {"type": "image", "url": "ignored"},
        ]
    assert _msg_text(Msg()) == "part one  part two "


def test_msg_text_missing_content_returns_empty():
    from axor_langchain.middleware import _msg_text
    class Msg: pass
    assert _msg_text(Msg()) == ""


def test_msg_tokens_minimum_one():
    from axor_langchain.middleware import _msg_tokens
    class Tiny: content = ""
    assert _msg_tokens(Tiny()) == 1
    class Short: content = "hi"
    assert _msg_tokens(Short()) == 1  # 2 chars // 4 = 0 → max(0, 1) = 1


def test_msg_tokens_estimates_four_chars_per_token():
    from axor_langchain.middleware import _msg_tokens
    class M: content = "x" * 40
    assert _msg_tokens(M()) == 10


def test_tool_name_from_dict():
    from axor_langchain.middleware import _tool_name
    assert _tool_name({"name": "calc"}) == "calc"
    assert _tool_name({}) == "unknown"


def test_tool_name_from_object():
    from axor_langchain.middleware import _tool_name
    class T: name = "search"
    assert _tool_name(T()) == "search"


def test_tool_call_id_from_dict_and_object():
    from axor_langchain.middleware import _tool_call_id
    assert _tool_call_id({"id": "abc"}) == "abc"
    assert _tool_call_id({}) == ""
    class T: id = "xyz"
    assert _tool_call_id(T()) == "xyz"


def test_resolve_telemetry_mode_kwarg_wins(monkeypatch):
    from axor_langchain.middleware import _resolve_telemetry_mode
    monkeypatch.setenv("AXOR_TELEMETRY", "remote")
    assert _resolve_telemetry_mode("local") == "local"
    assert _resolve_telemetry_mode("LOCAL") == "local"


def test_resolve_telemetry_mode_falls_back_to_env(monkeypatch):
    from axor_langchain.middleware import _resolve_telemetry_mode
    monkeypatch.setenv("AXOR_TELEMETRY", "REMOTE")
    assert _resolve_telemetry_mode(None) == "remote"


def test_resolve_telemetry_mode_defaults_off(monkeypatch):
    from axor_langchain.middleware import _resolve_telemetry_mode
    monkeypatch.delenv("AXOR_TELEMETRY", raising=False)
    assert _resolve_telemetry_mode(None) == "off"


def test_build_telemetry_pipeline_off_returns_none():
    from axor_langchain.middleware import _build_telemetry_pipeline
    assert _build_telemetry_pipeline(mode="off", axor_version="0.1.0") is None


def test_build_telemetry_pipeline_invalid_mode_returns_none():
    from axor_langchain.middleware import _build_telemetry_pipeline
    assert _build_telemetry_pipeline(mode="garbage", axor_version="") is None
