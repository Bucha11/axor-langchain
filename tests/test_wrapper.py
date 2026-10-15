"""
Tests for AxorToolWrapper — spec §7.4.

Tests:
(1) callback omitted → wrapper still denies
(2) raw tool reference not in wrapper's public API
(3) sync deny before original tool runs
(4) async deny before original tool runs
(5) callback-only mode (no policy_fn) logs observability-only warning
(6) original tool unreachable through wrapper public API
"""
from __future__ import annotations

import logging
import pytest
from typing import Any

from langchain_core.tools import BaseTool


class _RecordingTool(BaseTool):
    """Test tool that records whether it was called."""
    name: str = "recording_tool"
    description: str = "Records calls"
    called: bool = False
    call_args: dict = {}

    def _run(self, *args: Any, **kwargs: Any) -> str:
        self.called = True
        self.call_args = kwargs or {}
        return "tool_executed"

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        self.called = True
        self.call_args = kwargs or {}
        return "tool_executed_async"


def _deny_all(normalized) -> str:
    return "test denial reason"


def _allow_all(normalized) -> str | None:
    return None


async def _async_allow_all(normalized) -> str | None:
    return None


def _make_wrapper(policy_fn=_deny_all, tool=None, obs_callback=None):
    from axor_langchain.wrapper import AxorToolWrapper
    if tool is None:
        tool = _RecordingTool()
    return AxorToolWrapper(tool=tool, policy_fn=policy_fn, observability_callback=obs_callback)


# ── (1) callback omitted → wrapper still denies ────────────────────────────────

def test_wrapper_denies_without_observability_callback():
    """Enforcement is independent of observability callback."""
    wrapper = _make_wrapper(policy_fn=_deny_all, obs_callback=None)
    result = wrapper._run()
    assert "tool_denied" in result
    assert "test denial reason" in result


# ── (2) raw tool reference not in wrapper's public API ─────────────────────────

def test_raw_tool_not_in_public_api():
    """The wrapped tool must not be accessible via public fields."""
    from axor_langchain.wrapper import AxorToolWrapper
    tool = _RecordingTool()
    wrapper = _make_wrapper(tool=tool)
    # Public attributes should not expose the raw tool
    public_attrs = {k: v for k, v in vars(wrapper).items() if not k.startswith("_")}
    for v in public_attrs.values():
        assert v is not tool, f"raw tool exposed via public attribute"
    # model_fields (Pydantic) should not expose the raw tool
    for field_name in type(wrapper).model_fields:
        val = getattr(wrapper, field_name, None)
        assert val is not tool, f"raw tool exposed via model_field {field_name!r}"


# ── (3) sync deny before original tool runs ────────────────────────────────────

def test_sync_deny_before_original_tool_runs():
    """Policy denial must stop execution before the original tool's _run is called."""
    recording_tool = _RecordingTool()
    wrapper = _make_wrapper(policy_fn=_deny_all, tool=recording_tool)
    result = wrapper._run()
    assert "tool_denied" in result
    assert not recording_tool.called, "original tool must not have been called"


def test_sync_allow_calls_original_tool():
    """When policy allows, the original tool's _run is called."""
    recording_tool = _RecordingTool()
    wrapper = _make_wrapper(policy_fn=_allow_all, tool=recording_tool)
    result = wrapper._run()
    assert result == "tool_executed"
    assert recording_tool.called


# ── (4) async deny before original tool runs ───────────────────────────────────

@pytest.mark.asyncio
async def test_async_deny_before_original_tool_runs():
    """Async policy denial must stop execution before _arun is called."""
    recording_tool = _RecordingTool()
    wrapper = _make_wrapper(policy_fn=_deny_all, tool=recording_tool)
    result = await wrapper._arun()
    assert "tool_denied" in result
    assert not recording_tool.called, "original tool must not have been called"


@pytest.mark.asyncio
async def test_async_allow_calls_original_tool():
    """When policy allows, the original tool's _arun is called."""
    recording_tool = _RecordingTool()
    wrapper = _make_wrapper(policy_fn=_allow_all, tool=recording_tool)
    result = await wrapper._arun()
    assert result == "tool_executed_async"
    assert recording_tool.called


# ── (5) no policy_fn: the kernel gate still decides ────────────────────────────

def test_no_shared_gate_logs_warning_and_still_gates(caplog):
    """Without policy_fn the call still goes through the kernel gate; without a shared
    gate the wrapper says its ledger is its own."""
    from axor_langchain.wrapper import AxorToolWrapper
    with caplog.at_level(logging.WARNING, logger="axor.langchain.wrapper"):
        wrapper = AxorToolWrapper(tool=_RecordingTool(), policy_fn=None)
    assert any("without a shared gate" in msg for msg in caplog.messages)
    assert wrapper._run() == "tool_executed"


def test_async_policy_fn_rejected():
    from axor_langchain.wrapper import AxorToolWrapper
    with pytest.raises(TypeError, match="policy_fn must be synchronous"):
        AxorToolWrapper(tool=_RecordingTool(), policy_fn=_async_allow_all)


def test_no_policy_fn_allows_all():
    """Without policy_fn, no enforcement — all calls pass through."""
    recording_tool = _RecordingTool()
    from axor_langchain.wrapper import AxorToolWrapper
    wrapper = AxorToolWrapper(tool=recording_tool, policy_fn=None)
    result = wrapper._run()
    assert recording_tool.called
    assert result == "tool_executed"


# ── (6) original tool unreachable through wrapper public API ───────────────────

def test_original_tool_unreachable_through_public_api():
    """
    The original tool object should not be reachable via any public attribute
    of the wrapper. Private attributes (starting with '_') are acceptable.
    """
    from axor_langchain.wrapper import AxorToolWrapper
    tool = _RecordingTool()
    wrapper = AxorToolWrapper(tool=tool, policy_fn=_deny_all)

    # Walk all public attributes (no leading '_')
    def _find_tool(obj, depth=0, visited=None):
        if depth > 3:
            return False
        if visited is None:
            visited = set()
        obj_id = id(obj)
        if obj_id in visited:
            return False
        visited.add(obj_id)
        if obj is tool:
            return True
        if isinstance(obj, dict):
            for v in obj.values():
                if _find_tool(v, depth + 1, visited):
                    return True
        return False

    for attr in dir(wrapper):
        if attr.startswith("_"):
            continue  # private attrs are acceptable
        try:
            val = getattr(wrapper, attr)
        except Exception:
            continue
        if callable(val):
            continue  # methods are fine
        assert not _find_tool(val), (
            f"original tool reachable via public attribute {attr!r}"
        )


# ── Observability callback is called ───────────────────────────────────────────

def test_observability_callback_called_on_deny():
    """Observability callback is called even when policy denies."""
    calls = []
    def obs(tool_name, normalized, denial):
        calls.append((tool_name, denial))

    wrapper = _make_wrapper(policy_fn=_deny_all, obs_callback=obs)
    wrapper._run()
    assert len(calls) == 1
    assert calls[0][0] == "recording_tool"
    assert calls[0][1] == "test denial reason"


def test_observability_callback_called_on_allow():
    """Observability callback is called when policy allows."""
    calls = []
    def obs(tool_name, normalized, denial):
        calls.append((tool_name, denial))

    wrapper = _make_wrapper(policy_fn=_allow_all, obs_callback=obs)
    wrapper._run()
    assert len(calls) == 1
    assert calls[0][1] is None  # no denial
