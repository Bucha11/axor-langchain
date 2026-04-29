"""
Tool-result deduplication via (tool_name, args, content) match.

Production agents often re-call the same read-only tool with identical
args across turns. Without dedup, each repeat occupies the prompt
verbatim (or truncated) every model call. With dedup, the *older*
duplicate's content is replaced by a short pointer to the first
occurrence's tool_call_id; the recent-window copy stays verbatim so the
model can still reason on fresh data.

Contract:
  • OFF by default. The `aggressive` profile enables it.
  • Recent-window tool messages are NEVER deduplicated.
  • First occurrence of (name, args, content) is preserved verbatim.
  • Subsequent occurrences (outside recent window) AND only if the
    returned content matches the first occurrence get a short ref
    string. Same-args-but-different-content is NOT a duplicate —
    time-varying tools (metrics, logs, tickets, files after edits)
    must keep their fresh result.
  • Subsequent occurrences (outside recent window) get a short ref
    string, NOT total deletion — pairing with their AI tool_call must
    survive for provider validation.
  • Different args ≠ duplicate.
  • Different tool name ≠ duplicate.
"""
from __future__ import annotations

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from axor_langchain.middleware import (
    AxorMiddleware,
    _build_dedup_overrides,
    _msg_text,
)


def _round(tc_id: str, name: str, args: dict, output: str):
    return [
        AIMessage(content="", tool_calls=[
            {"name": name, "args": args, "id": tc_id}]),
        ToolMessage(content=output, tool_call_id=tc_id),
    ]


# ── _build_dedup_overrides (pure helper) ──────────────────────────────────────

class TestBuildDedupOverrides:

    def test_no_duplicates_no_overrides(self):
        msgs = [
            HumanMessage(content="task"),
            *_round("tc1", "search", {"q": "a"}, "result a"),
            *_round("tc2", "search", {"q": "b"}, "result b"),
            *_round("tc3", "fetch",  {"url": "x"}, "result x"),
        ]
        assert _build_dedup_overrides(msgs, recent_tools_window=1) == {}

    def test_old_duplicate_gets_pointer(self):
        msgs = [
            HumanMessage(content="task"),
            *_round("tc1", "search", {"q": "x"}, "RESULT"),
            *_round("tc2", "fetch",  {"url": "y"}, "y"),
            *_round("tc3", "search", {"q": "x"}, "RESULT"),  # old dup, same content
            *_round("tc4", "noop",   {}, "n"),  # most recent → window
        ]
        overrides = _build_dedup_overrides(msgs, recent_tools_window=1)
        # tc3 is the duplicate, tc4 is in recent window
        # message indices: 1,2 = round tc1; 3,4 = tc2; 5,6 = tc3; 7,8 = tc4
        assert 6 in overrides  # ToolMessage of tc3
        assert "tool_call_id=tc1" in overrides[6]
        assert 8 not in overrides  # tc4 in recent window stays verbatim

    def test_same_args_different_content_not_deduped(self):
        # Regression: time-varying tools (metrics, logs, file reads after
        # edits) can return different evidence for identical args. Such
        # results must NOT be replaced with a pointer to the prior one.
        msgs = [
            HumanMessage(content="task"),
            *_round("tc1", "query_metrics", {"window": "5m"}, "p99=120ms"),
            *_round("tc2", "noop", {}, "spacer"),
            *_round("tc3", "query_metrics", {"window": "5m"}, "p99=480ms"),
            *_round("tc4", "noop", {}, "fresh"),
        ]
        assert _build_dedup_overrides(msgs, recent_tools_window=1) == {}

    def test_recent_window_protects_duplicate(self):
        msgs = [
            HumanMessage(content="task"),
            *_round("tc1", "search", {"q": "x"}, "FIRST"),
            *_round("tc2", "search", {"q": "x"}, "DUP IN WINDOW"),
        ]
        # window=2 protects the last 2 ToolMessages — both, in this case.
        overrides = _build_dedup_overrides(msgs, recent_tools_window=2)
        assert overrides == {}

    def test_zero_window_dedups_everything_after_first(self):
        msgs = [
            HumanMessage(content="task"),
            *_round("tc1", "search", {"q": "x"}, "RESULT"),
            *_round("tc2", "search", {"q": "x"}, "RESULT"),
            *_round("tc3", "search", {"q": "x"}, "RESULT"),
        ]
        overrides = _build_dedup_overrides(msgs, recent_tools_window=0)
        # message indices: 1,2 / 3,4 / 5,6 — tool messages at 2,4,6
        assert 2 not in overrides    # first call kept
        assert 4 in overrides
        assert 6 in overrides

    def test_args_order_invariant(self):
        msgs = [
            HumanMessage(content="task"),
            *_round("tc1", "f", {"a": 1, "b": 2}, "SAME"),
            *_round("tc2", "noop", {}, "n"),
            *_round("tc3", "f", {"b": 2, "a": 1}, "SAME"),  # same hash, same content
            *_round("tc4", "noop", {}, "fresh"),
        ]
        overrides = _build_dedup_overrides(msgs, recent_tools_window=1)
        assert 6 in overrides

    def test_different_args_not_duplicate(self):
        msgs = [
            HumanMessage(content="task"),
            *_round("tc1", "f", {"a": 1}, "FIRST"),
            *_round("tc2", "f", {"a": 2}, "DIFFERENT"),
            *_round("tc3", "noop", {}, "fresh"),
        ]
        assert _build_dedup_overrides(msgs, recent_tools_window=1) == {}

    def test_different_tool_name_not_duplicate(self):
        msgs = [
            HumanMessage(content="task"),
            *_round("tc1", "alpha", {"q": "x"}, "A"),
            *_round("tc2", "beta",  {"q": "x"}, "B"),
            *_round("tc3", "noop",  {},        "fresh"),
        ]
        assert _build_dedup_overrides(msgs, recent_tools_window=1) == {}


# ── Pipeline integration ──────────────────────────────────────────────────────

class _Req:
    def __init__(self, messages=None, system_message=None, tools=None):
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


def _drive(axor, msgs):
    axor._ensure_engines()
    captured = {}
    def handler(req):
        captured["req"] = req
        return _Resp()
    axor.wrap_model_call(_Req(messages=msgs), handler)
    return captured["req"]


class TestPipelineDedup:

    def _build_dup_run(self, big_payload: str):
        # Old duplicate + irrelevant + recent tool round.
        return [
            SystemMessage(content="sys"),
            HumanMessage(content="checkout latency incident review"),
            *_round("tc1", "fetch_trace", {"trace_id": "abc"}, big_payload),
            *_round("tc2", "noop",        {}, "spacer"),
            *_round("tc3", "fetch_trace", {"trace_id": "abc"}, big_payload),
            *_round("tc4", "noop",        {}, "spacer2"),
            *_round("tc5", "fresh_round", {}, "FRESH"),  # protected by window
        ]

    def test_dedup_off_keeps_payload_verbatim(self):
        big = "X" * 5_000
        msgs = self._build_dup_run(big)
        axor = AxorMiddleware()  # default: dedup off
        seen = _drive(axor, msgs)
        old_dup = next(
            m for m in seen.messages
            if getattr(m, "tool_call_id", None) == "tc3"
        )
        # Compression may still trim, but it should NOT be the dedup pointer.
        text = _msg_text(old_dup)
        assert "duplicate tool call" not in text

    def test_dedup_on_replaces_old_duplicate(self):
        big = "X" * 5_000
        msgs = self._build_dup_run(big)
        axor = AxorMiddleware(tool_dedup_old_results=True,
                              recent_tools_window=1)
        seen = _drive(axor, msgs)
        old_dup = next(
            m for m in seen.messages
            if getattr(m, "tool_call_id", None) == "tc3"
        )
        text = _msg_text(old_dup)
        assert "duplicate tool call" in text
        assert "tool_call_id=tc1" in text
        assert len(text) < 200  # short pointer, not the 5kB blob

    def test_recent_round_stays_verbatim(self):
        big = "X" * 5_000
        msgs = self._build_dup_run(big)
        axor = AxorMiddleware(tool_dedup_old_results=True,
                              recent_tools_window=2)
        seen = _drive(axor, msgs)
        # tc5 is the most recent tool round — protected.
        fresh = next(
            m for m in seen.messages
            if getattr(m, "tool_call_id", None) == "tc5"
        )
        assert _msg_text(fresh) == "FRESH"

    def test_aggressive_profile_enables_dedup(self):
        big = "X" * 5_000
        msgs = self._build_dup_run(big)
        axor = AxorMiddleware(optimization_profile="aggressive",
                              # disable relevance gating to isolate dedup
                              tool_selection="off")
        seen = _drive(axor, msgs)
        old_dup = next(
            m for m in seen.messages
            if getattr(m, "tool_call_id", None) == "tc3"
        )
        assert "duplicate tool call" in _msg_text(old_dup)

    def test_paired_ai_tool_call_preserved(self):
        """After dedup, the AIMessage carrying tc3's tool_call must remain
        adjacent to the (now short) ToolMessage. Provider APIs reject
        orphan tool_call/tool_result.
        """
        big = "X" * 5_000
        msgs = self._build_dup_run(big)
        axor = AxorMiddleware(tool_dedup_old_results=True,
                              recent_tools_window=1)
        seen = _drive(axor, msgs)
        types = [getattr(m, "type", None) for m in seen.messages]
        # The triple "ai → ai (tc3 call) → tool (tc3 dedup'd)" must hold.
        for i, m in enumerate(seen.messages):
            if (getattr(m, "type", None) == "tool"
                    and getattr(m, "tool_call_id", None) == "tc3"):
                # Find the AI tool_call carrying tc3
                prev_ai = next(
                    (k for k in range(i - 1, -1, -1)
                     if getattr(seen.messages[k], "type", None) == "ai"),
                    None,
                )
                assert prev_ai is not None
                tc_ids = {
                    tc.get("id") if isinstance(tc, dict)
                    else getattr(tc, "id", "")
                    for tc in getattr(seen.messages[prev_ai],
                                      "tool_calls", None) or []
                }
                assert "tc3" in tc_ids
                break
        else:
            assert False, "tc3 ToolMessage missing"
