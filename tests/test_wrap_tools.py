"""Tests for AxorMiddleware.wrap_tools() — Step 17."""
from __future__ import annotations

import logging
import pytest
from langchain_core.tools import BaseTool
from typing import Any


class _SimpleTool(BaseTool):
    name: str = "simple_tool"
    description: str = "A simple test tool"
    call_count: int = 0

    def _run(self, *args: Any, **kwargs: Any) -> str:
        self.call_count += 1
        return "executed"

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        return "executed_async"


def _middleware(**kwargs):
    from axor_langchain.middleware import _AxorGovernanceCore
    return _AxorGovernanceCore(**kwargs)


def test_wrap_tools_returns_axor_tool_wrappers():
    from axor_langchain.wrapper import AxorToolWrapper
    m = _middleware()
    tools = [_SimpleTool(), _SimpleTool(name="another")]
    wrapped = m.wrap_tools(tools)
    assert len(wrapped) == 2
    wrapper_cls = type(wrapped[0])
    # Check it's an AxorToolWrapper (via class hierarchy)
    assert issubclass(wrapper_cls, BaseTool)
    assert "AxorToolWrapper" in wrapper_cls.__name__


def test_wrap_tools_enforces_allowed_tools():
    m = _middleware(allowed_tools=["allowed_tool"])
    tool = _SimpleTool(name="denied_tool")
    wrapped_list = m.wrap_tools([tool])
    wrapper = wrapped_list[0]
    result = wrapper._run()
    assert "tool_denied" in result
    assert tool.call_count == 0


def test_wrap_tools_passes_allowed_tools():
    m = _middleware(allowed_tools=["simple_tool"])
    tool = _SimpleTool()
    wrapped_list = m.wrap_tools([tool])
    wrapper = wrapped_list[0]
    result = wrapper._run()
    assert result == "executed"
    assert tool.call_count == 1


def test_wrap_tools_enforces_denied_tools():
    m = _middleware(denied_tools=["simple_tool"])
    tool = _SimpleTool()
    wrapped_list = m.wrap_tools([tool])
    wrapper = wrapped_list[0]
    result = wrapper._run()
    assert "tool_denied" in result
    assert tool.call_count == 0


def test_wrap_tools_custom_policy_fn():
    """Custom policy_fn overrides allow/deny list defaults."""
    m = _middleware()
    tool = _SimpleTool()
    wrapped_list = m.wrap_tools([tool], policy_fn=lambda _: "custom denial")
    wrapper = wrapped_list[0]
    result = wrapper._run()
    assert "custom denial" in result
    assert tool.call_count == 0


def test_wrap_tools_sets_flag():
    """wrap_tools() marks the middleware as wrapper-mode so callback warning is suppressed."""
    m = _middleware()
    assert not m._wrap_tools_called
    m.wrap_tools([_SimpleTool()])
    assert m._wrap_tools_called


def test_callback_only_warning_emitted(caplog):
    """_warn_callback_only_once() emits warning when wrappers not used."""
    m = _middleware()
    with caplog.at_level(logging.WARNING, logger="axor.langchain"):
        m._warn_callback_only_once()
    assert any("observability only" in msg.lower() for msg in caplog.messages)


def test_callback_only_warning_emitted_only_once(caplog):
    """Warning fires at most once per session."""
    m = _middleware()
    with caplog.at_level(logging.WARNING, logger="axor.langchain"):
        m._warn_callback_only_once()
        m._warn_callback_only_once()
    warning_msgs = [msg for msg in caplog.messages if "observability only" in msg.lower()]
    assert len(warning_msgs) == 1


def test_callback_only_warning_suppressed_after_wrap_tools(caplog):
    """No warning when wrap_tools() was already called."""
    m = _middleware()
    m.wrap_tools([_SimpleTool()])
    with caplog.at_level(logging.WARNING, logger="axor.langchain"):
        m._warn_callback_only_once()
    warning_msgs = [msg for msg in caplog.messages if "observability only" in msg.lower()]
    assert len(warning_msgs) == 0
