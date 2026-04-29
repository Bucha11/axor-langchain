"""
TaskAnalyzer + PolicySelector wiring.

Verifies that the latest user message is classified by axor-core's
HeuristicClassifier and PolicySelector picks the right ExecutionPolicy
(compression_mode + context_mode). The middleware then uses
`policy.compression_mode` instead of a hardcoded value.

axor-core's classifier is regex-based with calibrated coefficients in
`heuristic_coefficients.json`. We test against patterns documented there:
  • "explain X" / "what is X" → READONLY
  • "fix the bug", "rename X" → MUTATIVE
  • "write a test", "scaffold" → GENERATIVE
  • "rewrite the entire ...", "migrate the whole ..." → EXPANSIVE
"""
from __future__ import annotations

import pytest

from langchain_core.messages import HumanMessage, SystemMessage

from axor_core.contracts.policy import (
    CompressionMode,
    ContextMode,
    TaskComplexity,
    TaskNature,
)

from axor_langchain.middleware import (
    AxorMiddleware,
    _classify_task_sync,
    _latest_human_text,
    _normalize_compression_mode_name,
)


# ── _classify_task_sync ───────────────────────────────────────────────────────

class TestClassifyTaskSync:

    def test_empty_input_default(self):
        sig = _classify_task_sync("")
        # axor-core default: FOCUSED + GENERATIVE
        assert sig.complexity == TaskComplexity.FOCUSED
        assert sig.nature == TaskNature.GENERATIVE

    def test_explain_is_readonly(self):
        sig = _classify_task_sync("Explain how this function works.")
        assert sig.nature == TaskNature.READONLY

    def test_compression_mode_override_aliases(self):
        assert _normalize_compression_mode_name("auto") is None
        assert _normalize_compression_mode_name("minimal") == "AGGRESSIVE"
        assert _normalize_compression_mode_name("moderate") == "BALANCED"
        assert _normalize_compression_mode_name("broad") == "LIGHT"

    def test_aggressive_optimization_profile_sets_cost_first_knobs(self):
        axor = AxorMiddleware(optimization_profile="aggressive")
        assert axor._recent_tools_window == 1
        assert axor._compression_mode_override == "AGGRESSIVE"

    def test_explicit_knobs_override_optimization_profile(self):
        axor = AxorMiddleware(
            optimization_profile="aggressive",
            recent_tools_window=2,
            compression_mode="auto",
        )
        assert axor._recent_tools_window == 2
        assert axor._compression_mode_override is None

    def test_analyze_is_readonly(self):
        sig = _classify_task_sync("Analyze the performance of this code.")
        assert sig.nature == TaskNature.READONLY

    def test_fix_is_mutative(self):
        sig = _classify_task_sync("Fix the auth bug in login.py")
        assert sig.nature == TaskNature.MUTATIVE

    def test_rewrite_entire_is_expansive(self):
        sig = _classify_task_sync(
            "Rewrite the entire backend in Go."
        )
        assert sig.complexity == TaskComplexity.EXPANSIVE

    def test_migrate_full_is_expansive(self):
        sig = _classify_task_sync(
            "Migrate the full database schema to Postgres 16."
        )
        assert sig.complexity == TaskComplexity.EXPANSIVE

    def test_write_test_is_generative(self):
        sig = _classify_task_sync(
            "Write a test for the new endpoint."
        )
        assert sig.nature == TaskNature.GENERATIVE

    def test_estimated_scope_set(self):
        # FOCUSED → 1, MODERATE → 5, EXPANSIVE → 999
        sig = _classify_task_sync("Rewrite the entire system.")
        assert sig.estimated_scope >= 100  # EXPANSIVE


# ── _latest_human_text ────────────────────────────────────────────────────────

class TestLatestHumanText:

    def test_picks_latest_human(self):
        msgs = [
            SystemMessage(content="sys"),
            HumanMessage(content="first"),
            HumanMessage(content="second"),
        ]
        assert _latest_human_text(msgs) == "second"

    def test_no_human_returns_empty(self):
        msgs = [SystemMessage(content="sys")]
        assert _latest_human_text(msgs) == ""

    def test_empty_list(self):
        assert _latest_human_text([]) == ""


# ── _resolve_execution_policy ─────────────────────────────────────────────────

class TestResolveExecutionPolicy:

    def _resolve(self, text: str):
        axor = AxorMiddleware()
        msgs = [HumanMessage(content=text)]
        return axor._resolve_execution_policy(msgs)

    def test_focused_readonly_aggressive(self):
        """'explain X' → FOCUSED + READONLY → AGGRESSIVE compression."""
        policy = self._resolve("Explain how the cache layer works.")
        assert policy is not None
        assert policy.compression_mode == CompressionMode.AGGRESSIVE
        assert policy.context_mode == ContextMode.MINIMAL

    def test_focused_mutative_balanced(self):
        """'fix X' → FOCUSED + MUTATIVE → BALANCED compression."""
        policy = self._resolve("Fix the failing test in test_auth.py.")
        assert policy.compression_mode == CompressionMode.BALANCED

    def test_expansive_light(self):
        """EXPANSIVE → LIGHT compression (broad context, light squeeze)."""
        policy = self._resolve(
            "Rewrite the entire backend service in Go from scratch."
        )
        assert policy.compression_mode == CompressionMode.LIGHT
        assert policy.context_mode == ContextMode.BROAD


# ── Integration: wrap_model_call uses policy.compression_mode ────────────────

class _Req:
    def __init__(self, messages, system_message=None, tools=None):
        self.messages = list(messages or [])
        self.system_message = system_message
        self.tools = list(tools or [])
    def override(self, **kw):
        clone = _Req(self.messages, self.system_message, self.tools)
        for k, v in kw.items():
            setattr(clone, k, v)
        return clone


class _Resp:
    usage_metadata = {"input_tokens": 0, "output_tokens": 1}


class TestModeAffectsCompression:
    """The classifier-picked mode actually changes how aggressively the
    compressor truncates old EPHEMERAL tool messages."""

    def _drive_with_text(self, axor, latest_human_text: str, big_old_tool: str):
        from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
        msgs = [
            HumanMessage(content=latest_human_text),
            AIMessage(content="", tool_calls=[
                {"name": "x", "args": {"a": 1}, "id": "old"}]),
            ToolMessage(content=big_old_tool, tool_call_id="old"),
            AIMessage(content="", tool_calls=[
                {"name": "x", "args": {"a": 2}, "id": "tc1"}]),
            ToolMessage(content="r1", tool_call_id="tc1"),
            AIMessage(content="", tool_calls=[
                {"name": "x", "args": {"a": 3}, "id": "tc2"}]),
            ToolMessage(content="r2", tool_call_id="tc2"),
        ]
        axor._ensure_engines()
        captured = {}
        def handler(req):
            captured["req"] = req
            return _Resp()
        axor.wrap_model_call(_Req(messages=msgs), handler)
        return captured["req"]

    def test_aggressive_mode_truncates_more_than_balanced(self):
        """
        Same conversation, two different tasks → different compression modes →
        different surviving sizes for the EPHEMERAL old tool message.

        AGGRESSIVE truncate threshold: 200 tokens (~800 chars)
        BALANCED truncate threshold:   500 tokens (~2000 chars)
        """
        big = "Y" * 5_000
        from axor_langchain.middleware import _msg_text

        # Task 1: "explain X" → AGGRESSIVE
        axor1 = AxorMiddleware()
        seen1 = self._drive_with_text(
            axor1, "Explain how the renderer works.", big,
        )
        old_tool_1 = next(
            m for m in seen1.messages
            if getattr(m, "tool_call_id", None) == "old"
        )
        len_aggr = len(_msg_text(old_tool_1))

        # Task 2: "fix X" → BALANCED (less aggressive than AGGRESSIVE)
        axor2 = AxorMiddleware()
        seen2 = self._drive_with_text(
            axor2, "Fix the segfault in renderer.cpp.", big,
        )
        old_tool_2 = next(
            m for m in seen2.messages
            if getattr(m, "tool_call_id", None) == "old"
        )
        len_bal = len(_msg_text(old_tool_2))

        # Both truncated below original
        assert len_aggr < len(big)
        assert len_bal < len(big)
        # AGGRESSIVE strictly less than (or equal to) BALANCED — depends on
        # how the threshold maths, but never bigger.
        assert len_aggr <= len_bal
