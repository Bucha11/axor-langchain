from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage


def test_public_api_importable() -> None:
    from axor_langchain import AxorMiddleware, ToolStats, ToolCallStats, __version__

    assert AxorMiddleware is not None
    assert ToolStats is not None
    assert ToolCallStats is not None
    assert isinstance(__version__, str) and __version__


def test_middleware_instantiates_with_defaults() -> None:
    from axor_langchain import AxorMiddleware

    mw = AxorMiddleware()

    for hook in ("before_model", "wrap_model_call", "wrap_tool_call", "after_agent", "aafter_agent"):
        assert hasattr(mw, hook), f"missing hook: {hook}"


def test_middleware_accepts_governance_config() -> None:
    from axor_langchain import AxorMiddleware

    mw = AxorMiddleware(
        soft_token_limit=10_000,
        hard_token_limit=20_000,
        allowed_tools=["calculator"],
        denied_tools=["shell"],
        bypass_token_threshold=2_000,
    )
    assert mw is not None


def test_tool_call_stats_effective_rate() -> None:
    from axor_langchain import ToolCallStats

    fresh = ToolCallStats()
    assert fresh.effective_rate == 1.0
    assert fresh.success_rate == 1.0
    assert fresh.avg_latency_ms == 0.0

    mixed = ToolCallStats(call_count=8, error_count=2, denied_count=2, total_latency_ms=400.0)
    assert mixed.avg_latency_ms == 50.0
    assert mixed.success_rate == 0.75
    assert mixed.effective_rate == 0.6  # (8-2)/(8+2)


def test_tool_stats_aggregates_per_tool() -> None:
    from axor_langchain import ToolStats

    stats = ToolStats()
    stats.record_call("calc", 10.0)
    stats.record_call("calc", 30.0)
    stats.record_error("calc")
    stats.record_denied("shell")

    assert stats.total_calls == 2
    assert stats.total_errors == 1
    assert stats.by_tool["calc"].avg_latency_ms == 20.0
    assert stats.by_tool["shell"].denied_count == 1
    assert "calc" in stats.summary()


def test_messages_to_fragments_typed_by_role() -> None:
    from axor_langchain.middleware import _messages_to_fragments

    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="What is 2+2?"),
        AIMessage(content="4"),
        HumanMessage(content="And 3+3?"),
    ]

    fragments = _messages_to_fragments(messages)
    assert len(fragments) == len(messages)

    values = [getattr(f, "value", None) for f in fragments]
    assert str(values[0]).endswith("pinned") or values[0] == "pinned"  # system
    assert all(v is not None for v in values)


def test_tool_pair_repair_keeps_ai_before_tool_message() -> None:
    from axor_langchain.middleware import _repair_tool_pairs

    ai_with_call = AIMessage(content="", tool_calls=[{"name": "calc", "args": {"x": 1}, "id": "t1"}])
    tool_reply = ToolMessage(content="1", tool_call_id="t1")

    originals = [HumanMessage(content="go"), ai_with_call, tool_reply]
    repaired = _repair_tool_pairs([tool_reply], originals)

    types = [m.type for m in repaired]
    assert types == ["ai", "tool"]
