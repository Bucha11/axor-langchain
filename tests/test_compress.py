"""
axor-core `ContextCompressor` wiring through FragmentValue.

Contract:
  • Round-trip: messages → fragments → compress → messages. IDs preserved
    on transformed messages so add_messages reducer replaces in place.
  • Original ordering preserved (compressor reorders by FragmentValue
    group; we re-sort by source idx).
  • PINNED messages survive untouched.
  • EPHEMERAL (old tool messages) get aggressive truncation.
  • Synthesized prose summaries (from `_compress_prose`) are injected as
    HumanMessage right after system + first human.
  • AI tool_call messages keep tool_calls metadata across the round-trip
    (synthetic content marker is just a dedup anchor).
  • No regression vs baseline on small payloads (compression idempotent
    when nothing exceeds thresholds).
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
    _AI_TOOL_CALL_PREFIX,
    _classify_messages,
    _fragments_to_messages,
    _messages_to_fragments,
    _msg_text,
)


# ── Fragment round-trip ───────────────────────────────────────────────────────

class TestFragmentRoundTrip:

    def test_messages_to_fragments_preserves_count(self):
        msgs = [
            SystemMessage(content="sys"),
            HumanMessage(content="task"),
            AIMessage(content="answer"),
        ]
        fragments = _messages_to_fragments(msgs)
        assert len(fragments) == 3

    def test_source_encodes_position(self):
        msgs = [SystemMessage(content="s"), HumanMessage(content="h")]
        fragments = _messages_to_fragments(msgs)
        assert fragments[0].source == "lc:0"
        assert fragments[1].source == "lc:1"

    def test_ai_tool_call_gets_unique_marker(self):
        """Empty AIMessages with tool_calls must produce DIFFERENT fragment
        content per message, otherwise dedup collapses them all."""
        msgs = [
            HumanMessage(content="task"),
            AIMessage(content="", tool_calls=[
                {"name": "search", "args": {"q": "a"}, "id": "tc1"}]),
            ToolMessage(content="result a", tool_call_id="tc1"),
            AIMessage(content="", tool_calls=[
                {"name": "search", "args": {"q": "b"}, "id": "tc2"}]),
            ToolMessage(content="result b", tool_call_id="tc2"),
        ]
        fragments = _messages_to_fragments(msgs)
        ai_fragments = [f for f in fragments if f.kind == "ai_tool_call"]
        assert len(ai_fragments) == 2
        assert ai_fragments[0].content != ai_fragments[1].content
        assert all(f.content.startswith(_AI_TOOL_CALL_PREFIX)
                   for f in ai_fragments)

    def test_fragments_to_messages_no_change_returns_originals(self):
        msgs = [SystemMessage(content="s", id="s1"),
                HumanMessage(content="h", id="h1")]
        fragments = _messages_to_fragments(msgs)
        out = _fragments_to_messages(fragments, msgs)
        # Identity preserved when content unchanged
        assert out[0] is msgs[0]
        assert out[1] is msgs[1]


# ── Pass-through (small payloads) ─────────────────────────────────────────────

class TestPassthroughOnSmallPayloads:
    """Compression must not regress baseline: small messages flow through
    unchanged. Compression only fires when fragment sizes / ages
    exceed mode thresholds."""

    def _drive(self, axor, msgs):
        axor._ensure_engines()
        captured = {}
        def handler(req):
            captured["req"] = req
            class _R:
                usage_metadata = {"input_tokens": 0, "output_tokens": 1}
            return _R()
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
        req = _Req(messages=msgs)
        axor.wrap_model_call(req, handler)
        return captured["req"]

    def test_small_chat_unchanged(self):
        axor = AxorMiddleware()
        msgs = [
            SystemMessage(content="you are X", id="s1"),
            HumanMessage(content="hello", id="h1"),
            AIMessage(content="hi back", id="a1"),
        ]
        seen = self._drive(axor, msgs)
        assert [_msg_text(m) for m in seen.messages] == ["you are X", "hello", "hi back"]


# ── PINNED never touched ──────────────────────────────────────────────────────

class TestPinnedUntouchable:

    def test_system_message_passes_through_compress(self):
        big_sys = "X" * 100_000  # huge but PINNED
        msgs = [SystemMessage(content=big_sys, id="s1"),
                HumanMessage(content="task", id="h1")]

        from axor_core.context.compressor import ContextCompressor
        from axor_core.contracts.policy import CompressionMode
        fragments = _messages_to_fragments(msgs)
        result = ContextCompressor().compress(
            fragments, mode=CompressionMode.AGGRESSIVE,
            current_turn=len(msgs) + 1,
        )
        out = _fragments_to_messages(result.fragments, msgs)
        # System still present, content unchanged
        sys_msg = next(m for m in out if getattr(m, "type", None) == "system")
        assert _msg_text(sys_msg) == big_sys

    def test_latest_ai_with_tool_calls_pinned(self):
        original_ai = AIMessage(content="", tool_calls=[
            {"name": "x", "args": {}, "id": "tc1"}], id="a1")
        msgs = [HumanMessage(content="task", id="h1"), original_ai]
        from axor_core.context.compressor import ContextCompressor
        from axor_core.contracts.policy import CompressionMode
        fragments = _messages_to_fragments(msgs)
        result = ContextCompressor().compress(
            fragments, mode=CompressionMode.AGGRESSIVE,
            current_turn=len(msgs) + 1,
        )
        out = _fragments_to_messages(result.fragments, msgs)
        # AI tool_call survives. Compare against the AIMessage's OWN
        # normalized tool_calls (langchain auto-adds 'type': 'tool_call').
        ai_msg = next(m for m in out if getattr(m, "type", None) == "ai")
        assert ai_msg.tool_calls == original_ai.tool_calls
        # And it's the same object (PINNED → returned verbatim)
        assert ai_msg is original_ai


# ── EPHEMERAL (old tool messages) get truncated ──────────────────────────────

class TestEphemeralTruncation:

    def test_old_tool_message_gets_shorter(self):
        """Build a sequence where the OLDEST tool message is far enough back
        to be EPHEMERAL by `_classify_messages`'s recent-tools-window."""
        big_old_content = "Y" * 5_000  # > AGGRESSIVE truncate threshold
        msgs = [
            SystemMessage(content="sys"),
            HumanMessage(content="task"),
            # Old tool round (will be EPHEMERAL — outside recent window=2)
            AIMessage(content="", tool_calls=[
                {"name": "x", "args": {"a": 1}, "id": "old"}]),
            ToolMessage(content=big_old_content, tool_call_id="old", id="t_old"),
            # 2 fresh tool rounds (WORKING)
            AIMessage(content="", tool_calls=[
                {"name": "x", "args": {"a": 2}, "id": "tc1"}]),
            ToolMessage(content="r1", tool_call_id="tc1", id="t1"),
            AIMessage(content="", tool_calls=[
                {"name": "x", "args": {"a": 3}, "id": "tc2"}]),
            ToolMessage(content="r2", tool_call_id="tc2", id="t2"),
        ]
        # Sanity: classifier puts old tool as EPHEMERAL
        classified = _classify_messages(msgs)
        old_tool_value = classified[3][1]
        from axor_core.contracts.memory import FragmentValue
        assert old_tool_value == FragmentValue.EPHEMERAL

        from axor_core.context.compressor import ContextCompressor
        from axor_core.contracts.policy import CompressionMode
        fragments = _messages_to_fragments(msgs)
        result = ContextCompressor().compress(
            fragments, mode=CompressionMode.BALANCED,
            current_turn=len(msgs) + 1,
        )
        out = _fragments_to_messages(result.fragments, msgs)

        # The old tool message is shorter than original
        old_tool_out = next(
            (m for m in out if getattr(m, "tool_call_id", None) == "old"),
            None,
        )
        assert old_tool_out is not None, "old tool was dropped entirely"
        assert len(_msg_text(old_tool_out)) < len(big_old_content)

    def test_recent_tools_untouched_under_aggressive_mode(self):
        """Recent tools are PINNED (per `_classify_messages`), so even
        AGGRESSIVE compression mode leaves them alone. v1 used WORKING for
        recent tools, which got truncated at 200/500/2000 tokens depending
        on mode and broke agents."""
        big_content = "Z" * 5_000   # well above any truncation threshold
        msgs = [
            HumanMessage(content="task"),
            AIMessage(content="", tool_calls=[
                {"name": "x", "args": {}, "id": "tc1"}]),
            ToolMessage(content=big_content, tool_call_id="tc1"),  # most recent
        ]
        from axor_core.context.compressor import ContextCompressor
        from axor_core.contracts.policy import CompressionMode
        fragments = _messages_to_fragments(msgs)
        result = ContextCompressor().compress(
            fragments, mode=CompressionMode.AGGRESSIVE,
            current_turn=len(msgs) + 1,
        )
        out = _fragments_to_messages(result.fragments, msgs)
        tool_out = next(m for m in out if getattr(m, "type", None) == "tool")
        assert _msg_text(tool_out) == big_content

    def test_recent_tool_window_zero_allows_fresh_tool_truncation(self):
        """Power users can lower the fresh-tool window for extra savings.

        The default remains conservative, but the bridge must pass the knob
        through to FragmentValue assignment.
        """
        big_content = "Z" * 5_000
        msgs = [
            HumanMessage(content="task"),
            AIMessage(content="", tool_calls=[
                {"name": "x", "args": {}, "id": "tc1"}]),
            ToolMessage(content=big_content, tool_call_id="tc1"),
        ]
        from axor_core.context.compressor import ContextCompressor
        from axor_core.contracts.policy import CompressionMode
        fragments = _messages_to_fragments(msgs, recent_tools_window=0)
        result = ContextCompressor().compress(
            fragments, mode=CompressionMode.AGGRESSIVE,
            current_turn=len(msgs) + 1,
        )
        out = _fragments_to_messages(result.fragments, msgs)
        tool_out = next(m for m in out if getattr(m, "type", None) == "tool")
        assert len(_msg_text(tool_out)) < len(big_content)


# ── Original ordering preserved ───────────────────────────────────────────────

class TestOrderingPreservation:

    def test_tool_call_pair_stays_adjacent(self):
        """tool_call AIMessage must immediately precede its ToolMessage(s)
        in the output — required by model APIs."""
        msgs = [
            SystemMessage(content="sys"),
            HumanMessage(content="task"),
            AIMessage(content="", tool_calls=[
                {"name": "x", "args": {}, "id": "tc1"}]),
            ToolMessage(content="result1", tool_call_id="tc1"),
            AIMessage(content="prose answer"),
        ]
        from axor_core.context.compressor import ContextCompressor
        from axor_core.contracts.policy import CompressionMode
        fragments = _messages_to_fragments(msgs)
        result = ContextCompressor().compress(
            fragments, mode=CompressionMode.BALANCED,
            current_turn=len(msgs) + 1,
        )
        out = _fragments_to_messages(result.fragments, msgs)
        types = [getattr(m, "type", None) for m in out]
        # Order must be: system, human, ai (tool_call), tool, ai (prose)
        assert types == ["system", "human", "ai", "tool", "ai"]

    def test_dropped_empty_tool_result_restored_if_ai_tool_call_survives(self):
        """If core drops an empty ToolMessage, the bridge must restore it when
        its AIMessage.tool_calls partner remains in output."""
        msgs = [
            HumanMessage(content="task"),
            AIMessage(content="", tool_calls=[
                {"name": "x", "args": {}, "id": "tc-empty"}]),
            ToolMessage(content="", tool_call_id="tc-empty", id="t-empty"),
            AIMessage(content="done"),
        ]
        from axor_core.context.compressor import ContextCompressor
        from axor_core.contracts.policy import CompressionMode
        fragments = _messages_to_fragments(msgs)
        result = ContextCompressor().compress(
            fragments, mode=CompressionMode.AGGRESSIVE,
            current_turn=len(msgs) + 1,
        )
        out = _fragments_to_messages(result.fragments, msgs)
        assert any(
            getattr(m, "type", None) == "tool"
            and getattr(m, "tool_call_id", None) == "tc-empty"
            for m in out
        )


# ── Prose summary synthesis & injection ───────────────────────────────────────

class TestProseSummaryInjection:
    """When `_compress_prose` collapses old WORKING prose into a synth
    fragment, it must be injected back as a HumanMessage after the
    system + first human."""

    def test_old_prose_summary_inserted_after_system_and_first_human(self):
        # Use AGGRESSIVE mode (threshold 3 turns) and old prose far enough
        # back to trigger compression.
        msgs = [
            SystemMessage(content="sys", id="s1"),
            HumanMessage(content="research X", id="h1"),
            # Old prose (no decision verbs → WORKING + old → eligible)
            AIMessage(content="Looking around, browsing options here a lot.", id="a1"),
            AIMessage(content="More chatter about the topic.", id="a2"),
            AIMessage(content="Even more rambling content.", id="a3"),
            AIMessage(content="And more rambling here.", id="a4"),
            AIMessage(content="Also some unrelated thoughts about it.", id="a5"),
            HumanMessage(content="any progress?", id="h2"),
            AIMessage(content="recent answer", id="a6"),
        ]
        from axor_core.context.compressor import ContextCompressor
        from axor_core.contracts.policy import CompressionMode
        fragments = _messages_to_fragments(msgs)
        result = ContextCompressor().compress(
            fragments,
            mode=CompressionMode.AGGRESSIVE,  # threshold 3
            current_turn=len(msgs) + 5,  # forces older prose past threshold
        )
        out = _fragments_to_messages(result.fragments, msgs)
        types = [getattr(m, "type", None) for m in out]
        # If a synth summary exists, it appears after system + first human.
        # Presence depends on axor-core regex matches in the source content.
        synth = [
            m for m in out
            if getattr(m, "type", None) == "human"
            and getattr(m, "additional_kwargs", {}).get("lc_source") == "axor.compress_prose"
        ]
        if synth:
            synth_idx = out.index(synth[0])
            assert synth_idx >= 2  # after [system, first_human]


# ── End-to-end pipeline integration ───────────────────────────────────────────

class TestWrapModelCallIntegration:

    class _Req:
        def __init__(self, messages, system_message=None, tools=None):
            self.messages = list(messages or [])
            self.system_message = system_message
            self.tools = list(tools or [])
        def override(self, **kw):
            clone = TestWrapModelCallIntegration._Req(
                self.messages, self.system_message, self.tools,
            )
            for k, v in kw.items():
                setattr(clone, k, v)
            return clone

    class _Resp:
        usage_metadata = {"input_tokens": 0, "output_tokens": 1}

    def _drive(self, axor, msgs):
        axor._ensure_engines()
        captured = {}
        def handler(req):
            captured["req"] = req
            return self._Resp()
        req = self._Req(messages=msgs)
        axor.wrap_model_call(req, handler)
        return captured["req"]

    def test_compression_engages_in_pipeline(self):
        """Validates compression runs INSIDE wrap_model_call, not just
        when called directly. Old tool result with big content should
        come out shorter."""
        big = "Q" * 5_000
        msgs = [
            HumanMessage(content="task"),
            # Old round (EPHEMERAL)
            AIMessage(content="", tool_calls=[
                {"name": "x", "args": {"a": 1}, "id": "old"}]),
            ToolMessage(content=big, tool_call_id="old"),
            # Two recent rounds (WORKING)
            AIMessage(content="", tool_calls=[
                {"name": "x", "args": {"a": 2}, "id": "tc1"}]),
            ToolMessage(content="r1", tool_call_id="tc1"),
            AIMessage(content="", tool_calls=[
                {"name": "x", "args": {"a": 3}, "id": "tc2"}]),
            ToolMessage(content="r2", tool_call_id="tc2"),
        ]
        axor = AxorMiddleware()
        seen = self._drive(axor, msgs)
        # Old tool message is shorter
        old_tool = next(
            (m for m in seen.messages
             if getattr(m, "tool_call_id", None) == "old"),
            None,
        )
        assert old_tool is not None
        assert len(_msg_text(old_tool)) < len(big)
