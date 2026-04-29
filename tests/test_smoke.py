from __future__ import annotations


def test_public_api_importable() -> None:
    from axor_langchain import (
        AxorMiddleware,
        ToolCallStats,
        ToolStats,
        __version__,
    )

    assert AxorMiddleware is not None
    assert ToolStats is not None
    assert ToolCallStats is not None
    assert isinstance(__version__, str) and __version__


def test_middleware_instantiates_with_defaults() -> None:
    from axor_langchain import AxorMiddleware

    mw = AxorMiddleware()

    #  no `before_model` hook — compression moved to
    # `wrap_model_call`. The remaining hooks are still expected.
    for hook in (
        "wrap_model_call",
        "wrap_tool_call",
        "after_agent",
        "aafter_agent",
    ):
        assert hasattr(mw, hook), f"missing hook: {hook}"


def test_middleware_accepts_governance_config() -> None:
    from axor_langchain import AxorMiddleware

    mw = AxorMiddleware(
        soft_token_limit=10_000,
        hard_token_limit=20_000,
        allowed_tools=["calculator"],
        denied_tools=["shell"],
    )
    assert mw is not None


def test_tool_call_stats_effective_rate() -> None:
    from axor_langchain import ToolCallStats

    fresh = ToolCallStats()
    assert fresh.effective_rate == 1.0
    assert fresh.success_rate == 1.0
    assert fresh.avg_latency_ms == 0.0

    mixed = ToolCallStats(
        call_count=8, error_count=2, denied_count=2, total_latency_ms=400.0
    )
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
