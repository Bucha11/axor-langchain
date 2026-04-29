"""
Pass-through baseline + budget-recording invariants.

Contract: with default AxorMiddleware, the prompt the handler sees is
byte-equal to what the agent gave us (modulo personality and tool
filtering, both explicit opt-ins). Compression, budget gating, and
custom-state tests live elsewhere and assert their specific behavior.
"""

from __future__ import annotations

import pytest
from axor_langchain.middleware import (
    AxorMiddleware,
    _msg_text,
)

# ── Test doubles ──────────────────────────────────────────────────────────────


class FakeRequest:
    """ModelRequest stand-in. Supports override() like the real one."""

    def __init__(
        self,
        messages=None,
        system_message=None,
        tools=None,
        tool_choice=None,
        response_format=None,
        model_settings=None,
    ):
        self.messages = list(messages or [])
        self.system_message = system_message
        self.tools = list(tools or [])
        self.tool_choice = tool_choice
        self.response_format = response_format
        self.model_settings = model_settings or {}

    def override(self, **kwargs):
        clone = FakeRequest(
            messages=self.messages,
            system_message=self.system_message,
            tools=self.tools,
            tool_choice=self.tool_choice,
            response_format=self.response_format,
            model_settings=self.model_settings,
        )
        for k, v in kwargs.items():
            setattr(clone, k, v)
        return clone


class FakeResponse:
    def __init__(
        self,
        input_tokens=0,
        output_tokens=10,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
    ):
        self.usage_metadata = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_creation_input_tokens": cache_creation_input_tokens,
            "cache_read_input_tokens": cache_read_input_tokens,
        }


def fake_msg(type_: str, content):
    """Plain stub — no model_copy, falls into _merge_personality's
    AttributeError fallback path which creates a fresh SystemMessage."""

    class _Msg:
        pass

    m = _Msg()
    m.type = type_
    m.content = content
    return m


# ── Helper: drive wrap_model_call once on an AxorMiddleware instance ─────────


def _drive(axor, request, *, response_input=0, response_output=42):
    """
    Invoke axor.wrap_model_call with `request`, capturing the request the
    handler actually saw. Returns (handler_request, response).

    Provider-counted token usage is configurable so budget-tracking tests
    can assert against precise numbers (input_tokens are recorded post-call
    from response.usage_metadata, not pre-call estimation).
    """
    axor._ensure_engines()
    captured = {}

    def handler(req):
        captured["request"] = req
        return FakeResponse(
            input_tokens=response_input,
            output_tokens=response_output,
        )

    response = axor.wrap_model_call(request, handler)
    return captured["request"], response


async def _adrive(axor, request, *, response_input=0, response_output=42):
    axor._ensure_engines()
    captured = {}

    async def handler(req):
        captured["request"] = req
        return FakeResponse(
            input_tokens=response_input,
            output_tokens=response_output,
        )

    response = await axor.awrap_model_call(request, handler)
    return captured["request"], response


# ── Pass-through baseline ─────────────────────────────────────────────────────


class TestPassthroughBaseline:
    """
    Pass-through invariant: with an empty / default AxorMiddleware, the request
    seen by the handler is byte-equal to the request the agent gave us
    (modulo identity — override() returns a clone).
    """

    def test_messages_passed_through_unchanged(self):
        axor = AxorMiddleware()
        msgs = [fake_msg("human", "hello"), fake_msg("ai", "hi")]
        req = FakeRequest(messages=msgs)
        seen, _ = _drive(axor, req)
        assert [_msg_text(m) for m in seen.messages] == ["hello", "hi"]
        assert seen.system_message is None

    def test_system_message_passed_through_unchanged_when_no_personality(self):
        axor = AxorMiddleware()
        sys = fake_msg("system", "you are a careful agent")
        req = FakeRequest(system_message=sys)
        seen, _ = _drive(axor, req)
        # Same content; personality didn't run because it's None
        assert _msg_text(seen.system_message) == "you are a careful agent"

    def test_tools_passed_through_unchanged_when_no_filter(self):
        axor = AxorMiddleware()
        tools = [{"name": "search"}, {"name": "calc"}]
        req = FakeRequest(tools=tools)
        seen, _ = _drive(axor, req)
        assert [t["name"] for t in seen.tools] == ["search", "calc"]

    def test_no_extra_messages_added(self):
        axor = AxorMiddleware()
        msgs = [fake_msg("human", "task")]
        req = FakeRequest(messages=msgs)
        seen, _ = _drive(axor, req)
        assert len(seen.messages) == 1


# ── Personality merge (only when configured) ──────────────────────────────────


class TestPersonalityMerge:
    def test_personality_prepended_when_no_system(self):
        axor = AxorMiddleware(personality="be concise")
        req = FakeRequest()
        seen, _ = _drive(axor, req)
        assert _msg_text(seen.system_message) == "be concise"

    def test_personality_prepended_to_existing_system(self):
        axor = AxorMiddleware(personality="be concise")
        sys = fake_msg("system", "you are an agent")
        req = FakeRequest(system_message=sys)
        seen, _ = _drive(axor, req)
        text = _msg_text(seen.system_message)
        assert text.startswith("be concise")
        assert "you are an agent" in text

    def test_no_personality_no_change(self):
        axor = AxorMiddleware()  # personality=None
        sys = fake_msg("system", "original")
        req = FakeRequest(system_message=sys)
        seen, _ = _drive(axor, req)
        assert _msg_text(seen.system_message) == "original"


# ── Tool filtering ────────────────────────────────────────────────────────────


class TestToolFilter:
    def test_allowed_tools_whitelist(self):
        axor = AxorMiddleware(allowed_tools=["search"])
        req = FakeRequest(tools=[{"name": "search"}, {"name": "calc"}])
        seen, _ = _drive(axor, req)
        assert [t["name"] for t in seen.tools] == ["search"]

    def test_denied_tools_blacklist(self):
        axor = AxorMiddleware(denied_tools=["bash"])
        req = FakeRequest(tools=[{"name": "search"}, {"name": "bash"}])
        seen, _ = _drive(axor, req)
        assert [t["name"] for t in seen.tools] == ["search"]


# ── Budget tracking actually works ────────────────────────────────────────────


class TestBudgetRecording:
    """Two-phase accounting (post-rewrite):

    • Pre-call: estimate-based hard-limit gate. Raises BudgetExceededError
      when projected spend would exceed limit. Does NOT record anything.
    • Post-call: provider-counted input_tokens AND output_tokens both
      recorded from response.usage_metadata. Single authoritative write.
    """

    def test_provider_input_recorded_post_call(self):
        """Input tokens come from response.usage_metadata.input_tokens
        (provider count), not pre-call estimation. With a small message
        list and response reporting input_tokens=137, total should be
        exactly 137 + output."""
        axor = AxorMiddleware()
        msgs = [fake_msg("human", "X" * 200)]  # estimate would be ~50
        req = FakeRequest(messages=msgs)
        _drive(axor, req, response_input=137, response_output=10)
        # Provider-reported 137 is what goes in, not the estimated ~50
        assert axor.total_tokens_spent == 137 + 10

    def test_pre_call_estimate_does_not_record(self):
        """Pre-call estimate is for the gate only; no budget mutation
        happens until the response comes back."""
        axor = AxorMiddleware()
        msgs = [fake_msg("human", "X" * 4_000)]  # ~1000 tokens estimated
        req = FakeRequest(messages=msgs)
        # Response says only 5 tokens — the estimate of 1000 must NOT
        # leak into the tracker.
        _drive(axor, req, response_input=5, response_output=2)
        assert axor.total_tokens_spent == 5 + 2

    def test_output_tokens_recorded_from_response(self):
        axor = AxorMiddleware()
        req = FakeRequest(messages=[fake_msg("human", "hi")])
        before = axor.total_tokens_spent
        _drive(axor, req, response_input=3, response_output=42)
        # Response had output_tokens=42 + input_tokens=3, total grew by 45
        assert axor.total_tokens_spent == before + 45

    def test_prompt_cache_tokens_recorded_from_response(self):
        from axor_core.budget import TokenCostRates

        axor = AxorMiddleware()
        req = FakeRequest(messages=[fake_msg("human", "hi")])

        def handler(_req):
            return FakeResponse(
                input_tokens=10,
                output_tokens=5,
                cache_creation_input_tokens=100,
                cache_read_input_tokens=1000,
            )

        axor.wrap_model_call(req, handler)

        assert axor.total_tokens_spent == 1115

        priced = AxorMiddleware(
            token_cost_rates=TokenCostRates(input_per_m=3.0, output_per_m=15.0)
        )
        priced.wrap_model_call(req, handler)
        summary = priced.cost_summary()
        assert summary is not None
        assert summary["total_cost"] == pytest.approx(0.00078)

    def test_model_response_result_usage_recorded(self):
        """LangChain v1 wrap_model_call returns ModelResponse(result=[AIMessage]).
        Usage lives on the AIMessage, not on the ModelResponse itself."""
        from langchain.agents.middleware import ModelResponse
        from langchain_core.messages import AIMessage

        axor = AxorMiddleware()
        axor._ensure_engines()
        msg = AIMessage(
            content="ok",
            usage_metadata={
                "input_tokens": 17,
                "output_tokens": 5,
                "cache_creation_input_tokens": 100,
                "cache_read_input_tokens": 1000,
                "total_tokens": 22,
            },
        )
        axor._record_usage_from_response(ModelResponse(result=[msg]))
        assert axor.total_tokens_spent == 1122

    def test_hard_limit_raises_budget_exceeded(self):
        """Hard limit → BudgetExceededError raised BEFORE handler runs.
        Replaces the v1 silent `_cancelled` flag that was never enforced."""
        from axor_core.errors import BudgetExceededError

        axor = AxorMiddleware(soft_token_limit=1)  # hard_limit = 1.5
        big_msg = fake_msg("human", "X" * 10_000)  # ~2500 tokens estimate
        req = FakeRequest(messages=[big_msg])
        handler_called = []

        def handler(_req):
            handler_called.append(True)
            return FakeResponse(output_tokens=1)

        axor._ensure_engines()
        with pytest.raises(BudgetExceededError) as excinfo:
            axor.wrap_model_call(req, handler)
        # Handler MUST NOT have been invoked
        assert handler_called == []
        # Exception carries diagnostic data
        err = excinfo.value
        assert err.spent == 0
        assert err.projected > 0
        assert err.limit == 1  # soft 1 → hard 1.5 → int(1.5) = 1

    def test_no_hard_limit_no_raise(self):
        """When hard_token_limit isn't configured, the gate is a no-op
        (legacy / opt-in for cost-sensitive deployments)."""
        axor = AxorMiddleware()  # no limits at all
        big_msg = fake_msg("human", "X" * 10_000)
        req = FakeRequest(messages=[big_msg])
        # Should NOT raise
        _drive(axor, req)


# ── Async hook parity ─────────────────────────────────────────────────────────


class TestAsyncHooks:
    @pytest.mark.asyncio
    async def test_awrap_model_call_matches_sync_path(self):
        axor = AxorMiddleware(allowed_tools=["search"], personality="be brief")
        req = FakeRequest(
            messages=[fake_msg("human", "hello")],
            tools=[{"name": "search"}, {"name": "bash"}],
        )
        seen, _ = await _adrive(axor, req, response_input=11, response_output=7)
        assert [t["name"] for t in seen.tools] == ["search"]
        assert _msg_text(seen.system_message) == "be brief"
        assert axor.total_tokens_spent == 18


# ── AxorState extension is declared and importable ────────────────────────────


class TestAxorStateExposed:
    """AxorState extends AgentState with `tool_result_cache` so LangGraph
    checkpoints persist tool memo across invocations."""

    def test_axor_state_has_tool_result_cache(self):
        from axor_langchain.middleware import AxorState

        assert "tool_result_cache" in AxorState.__annotations__
        # Inherits messages from AgentState
        assert "messages" in AxorState.__annotations__

    def test_axor_state_via_package(self):
        # `from axor_langchain import AxorState` works
        import axor_langchain

        assert "tool_result_cache" in axor_langchain.AxorState.__annotations__
