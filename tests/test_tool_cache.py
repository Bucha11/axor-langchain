"""
Tool-result memoization via custom AgentState (`tool_result_cache`).

Contract:
  • Caching is opt-in per tool name. Same (tool_name, args) twice for an
    opted-in tool → second call returns cached content and skips handler.
  • Cache survives across invocations under the same thread_id (via
    LangGraph checkpointing, which is exercised by other layers — here
    we test the state-update mechanism that makes it possible).
  • Hash is order-invariant on dict args.
  • Errors are NOT cached.
  • Without AxorState wired (state has no `tool_result_cache` key), the
    middleware degrades gracefully — no cache hit, no error.
"""
from __future__ import annotations

import pytest

from langchain_core.messages import ToolMessage
from langgraph.types import Command

from axor_langchain.middleware import (
    AxorMiddleware,
    _hash_tool_call,
)


# ── Test doubles ──────────────────────────────────────────────────────────────

class FakeToolCallRequest:
    """ToolCallRequest stand-in with state attribute."""
    def __init__(self, tool_call: dict, state: dict | None = None):
        self.tool_call = tool_call
        self.state = state if state is not None else {}


def _drive_tool(axor, request, handler):
    """Invoke wrap_tool_call once. Eagerly init engines for budget recording."""
    axor._ensure_engines()
    return axor.wrap_tool_call(request, handler)


async def _adrive_tool(axor, request, handler):
    axor._ensure_engines()
    return await axor.awrap_tool_call(request, handler)


# ── _hash_tool_call ───────────────────────────────────────────────────────────

class TestHashToolCall:

    def test_dict_args_order_invariant(self):
        a = _hash_tool_call("read", {"path": "x.py", "lines": 10})
        b = _hash_tool_call("read", {"lines": 10, "path": "x.py"})
        assert a == b

    def test_different_args_different_hash(self):
        a = _hash_tool_call("read", {"path": "a.py"})
        b = _hash_tool_call("read", {"path": "b.py"})
        assert a != b

    def test_different_tool_different_hash(self):
        a = _hash_tool_call("read", {"path": "x"})
        b = _hash_tool_call("write", {"path": "x"})
        assert a != b

    def test_non_serializable_falls_back_to_repr(self):
        class Weird:
            pass
        h = _hash_tool_call("tool", {"thing": Weird()})
        assert isinstance(h, str)
        assert len(h) == 32

    def test_idempotent(self):
        # Same input → same hash on every call (required for cross-invocation
        # cache replay via checkpoint).
        a = _hash_tool_call("search", {"q": "EU energy"})
        b = _hash_tool_call("search", {"q": "EU energy"})
        assert a == b


# ── Cache miss runs handler and writes to state ───────────────────────────────

class TestCacheMiss:

    def test_handler_called_on_miss(self):
        axor = AxorMiddleware(cache_tools=["read_file"])
        calls = []
        def handler(req):
            calls.append(req.tool_call)
            return "FILE CONTENT"
        req = FakeToolCallRequest(
            tool_call={"name": "read_file", "args": {"path": "x.py"}, "id": "tc1"},
            state={"messages": [], "tool_result_cache": {}},
        )
        result = _drive_tool(axor, req, handler)
        assert len(calls) == 1
        # Returns Command with state update (because we wrote to cache)
        assert isinstance(result, Command)
        assert "tool_result_cache" in result.update
        # New cache contains the entry
        new_cache = result.update["tool_result_cache"]
        h = _hash_tool_call("read_file", {"path": "x.py"})
        assert h in new_cache
        assert new_cache[h] == "FILE CONTENT"
        # ToolMessage is in the update
        msgs = result.update["messages"]
        assert len(msgs) == 1
        assert msgs[0].content == "FILE CONTENT"
        assert msgs[0].tool_call_id == "tc1"


# ── Cache hit skips handler ───────────────────────────────────────────────────

class TestCacheHit:

    def test_handler_not_called_on_hit(self):
        axor = AxorMiddleware(cache_tools=["read_file"])
        h = _hash_tool_call("read_file", {"path": "x.py"})
        cache = {h: "PRELOADED CONTENT"}
        req = FakeToolCallRequest(
            tool_call={"name": "read_file", "args": {"path": "x.py"}, "id": "tc2"},
            state={"messages": [], "tool_result_cache": cache},
        )
        calls = []
        def handler(req):
            calls.append(req.tool_call)
            return "should not be called"
        result = _drive_tool(axor, req, handler)
        assert len(calls) == 0  # handler skipped entirely
        # Returns plain ToolMessage (no state update — cache wasn't modified)
        assert isinstance(result, ToolMessage)
        assert result.content == "PRELOADED CONTENT"
        assert result.tool_call_id == "tc2"

    def test_cache_hit_uses_current_tool_call_id(self):
        # Cached content was originally returned for tc1, but on the second
        # call the agent uses tc99. The new ToolMessage must carry tc99.
        axor = AxorMiddleware(cache_tools=["search"])
        h = _hash_tool_call("search", {"q": "EU"})
        req = FakeToolCallRequest(
            tool_call={"name": "search", "args": {"q": "EU"}, "id": "tc99"},
            state={"messages": [], "tool_result_cache": {h: "cached"}},
        )
        result = _drive_tool(axor, req, lambda r: "fresh")
        assert isinstance(result, ToolMessage)
        assert result.tool_call_id == "tc99"
        assert result.content == "cached"

    def test_cache_hit_records_stat(self):
        axor = AxorMiddleware(track_tool_stats=True, cache_tools=["search"])
        h = _hash_tool_call("search", {"q": "X"})
        req = FakeToolCallRequest(
            tool_call={"name": "search", "args": {"q": "X"}, "id": "tc1"},
            state={"messages": [], "tool_result_cache": {h: "cached"}},
        )
        _drive_tool(axor, req, lambda r: "no")
        stats = axor.tool_stats
        assert stats is not None
        assert stats.by_tool["search"].cache_hits == 1
        assert stats.by_tool["search"].call_count == 0
        assert stats.total_cache_hits == 1


# ── Round-trip: miss then hit ─────────────────────────────────────────────────

class TestRoundTrip:

    def test_miss_then_hit(self):
        """First call writes to cache; the SAME state passed through to the
        next call shows the second call as a hit. This is the in-memory
        equivalent of LangGraph checkpoint replay."""
        axor = AxorMiddleware(cache_tools=["search"])
        state = {"messages": [], "tool_result_cache": {}}
        tc = {"name": "search", "args": {"q": "axor"}, "id": "tc1"}

        # First call: miss
        req1 = FakeToolCallRequest(tool_call=tc, state=state)
        calls = []
        result1 = _drive_tool(
            axor, req1,
            lambda r: (calls.append("call1"), "RESULT 1")[1],
        )
        assert isinstance(result1, Command)
        # Apply the update to state (simulating LangGraph reducer)
        state["tool_result_cache"] = result1.update["tool_result_cache"]

        # Second call with same args: hit
        req2 = FakeToolCallRequest(
            tool_call={**tc, "id": "tc2"},  # different id, same args
            state=state,
        )
        result2 = _drive_tool(
            axor, req2,
            lambda r: (calls.append("call2"), "should not run")[1],
        )
        assert calls == ["call1"]  # only first handler ran
        assert isinstance(result2, ToolMessage)
        assert result2.content == "RESULT 1"

    @pytest.mark.asyncio
    async def test_async_miss_then_hit(self):
        axor = AxorMiddleware(cache_tools=["search"])
        state = {"messages": [], "tool_result_cache": {}}
        tc = {"name": "search", "args": {"q": "axor"}, "id": "tc1"}
        calls = []

        async def handler(req):
            calls.append(req.tool_call["id"])
            return "ASYNC RESULT"

        result1 = await _adrive_tool(axor, FakeToolCallRequest(tc, state), handler)
        assert isinstance(result1, Command)
        state["tool_result_cache"] = result1.update["tool_result_cache"]

        result2 = await _adrive_tool(
            axor,
            FakeToolCallRequest({**tc, "id": "tc2"}, state),
            handler,
        )
        assert calls == ["tc1"]
        assert isinstance(result2, ToolMessage)
        assert result2.content == "ASYNC RESULT"


# ── Errors are not cached ─────────────────────────────────────────────────────

class TestErrorNotCached:

    def test_error_handler_path_does_not_cache(self):
        # Handled tool errors return a ToolMessage and do not write cache state.
        axor = AxorMiddleware(
            tool_error_handler=lambda name, exc: f"[{name}: {exc}]",
            tool_max_retries=0,
        )

        def failing_handler(req):
            raise RuntimeError("kaboom")

        req = FakeToolCallRequest(
            tool_call={"name": "search", "args": {"q": "X"}, "id": "tc1"},
            state={"messages": [], "tool_result_cache": {}},
        )
        result = _drive_tool(axor, req, failing_handler)
        # Returns plain ToolMessage (error message), NOT a Command
        assert isinstance(result, ToolMessage)
        assert "kaboom" in result.content
        # No state update was emitted, so no cache pollution

    def test_successful_string_tool_caches_with_error_handler_configured(self):
        # Regression: string return + tool_error_handler must still write
        # cache. Earlier `isinstance(result, str) and handler is not None`
        # short-circuited *both* success and handled-error paths.
        axor = AxorMiddleware(
            cache_tools=["search"],
            tool_error_handler=lambda name, exc: f"[{name}: {exc}]",
            tool_max_retries=0,
        )
        req = FakeToolCallRequest(
            tool_call={"name": "search", "args": {"q": "Z"}, "id": "tc1"},
            state={"messages": [], "tool_result_cache": {}},
        )
        result = _drive_tool(axor, req, lambda r: "OK")
        assert isinstance(result, Command)
        assert result.update["tool_result_cache"], "successful tool must cache"

    @pytest.mark.asyncio
    async def test_successful_string_tool_caches_with_error_handler_async(self):
        axor = AxorMiddleware(
            cache_tools=["search"],
            tool_error_handler=lambda name, exc: f"[{name}: {exc}]",
            tool_max_retries=0,
        )

        async def ok_handler(req):
            return "OK"

        req = FakeToolCallRequest(
            tool_call={"name": "search", "args": {"q": "Z"}, "id": "tc1"},
            state={"messages": [], "tool_result_cache": {}},
        )
        result = await _adrive_tool(axor, req, ok_handler)
        assert isinstance(result, Command)
        assert result.update["tool_result_cache"]


# ── Graceful fallback when state has no tool_result_cache field ──────────────

class TestNoSchemaGracefulFallback:

    def test_state_without_tool_result_cache_works(self):
        """If a user forgets `state_schema=AxorState` (or langchain version
        doesn't honor it), state may not contain `tool_result_cache`. The
        middleware must NOT crash; it just doesn't memoize."""
        axor = AxorMiddleware(cache_tools=["search"])
        req = FakeToolCallRequest(
            tool_call={"name": "search", "args": {"q": "X"}, "id": "tc1"},
            state={"messages": []},  # no tool_result_cache field
        )
        result = _drive_tool(axor, req, lambda r: "RESULT")
        # Middleware initializes cache state even if LangGraph later drops it.
        assert isinstance(result, Command)
        assert "tool_result_cache" in result.update

    def test_none_state_works(self):
        """Edge: request.state is None (some test scaffolds)."""
        axor = AxorMiddleware(cache_tools=["search"])
        req = FakeToolCallRequest(
            tool_call={"name": "search", "args": {"q": "X"}, "id": "tc1"},
            state=None,
        )
        result = _drive_tool(axor, req, lambda r: "RESULT")
        assert isinstance(result, Command)


class TestDefaultNoCache:

    def test_default_executes_even_with_preloaded_cache(self):
        """Cache is disabled unless `cache_tools` explicitly includes the tool."""
        h = _hash_tool_call("write_file", {"path": "x.py"})
        axor = AxorMiddleware()
        req = FakeToolCallRequest(
            tool_call={"name": "write_file", "args": {"path": "x.py"}, "id": "tc1"},
            state={"messages": [], "tool_result_cache": {h: "STALE"}},
        )
        calls = []
        result = _drive_tool(
            axor,
            req,
            lambda r: (calls.append("called"), "FRESH")[1],
        )
        assert calls == ["called"]
        assert isinstance(result, ToolMessage)
        assert result.content == "FRESH"


# ── FIFO eviction ────────────────────────────────────────────────────────────

class TestFifoEviction:
    """`tool_result_cache` must stay bounded; LangGraph serializes the whole
    state every checkpoint, so an unbounded cache gives O(N) overhead per turn."""

    def _drive(self, axor, tool_name, args, prev_cache):
        req = FakeToolCallRequest(
            tool_call={"name": tool_name, "args": args, "id": f"tc-{args}"},
            state={"messages": [], "tool_result_cache": dict(prev_cache)},
        )
        result = _drive_tool(axor, req, lambda r: f"OUT-{args}")
        assert isinstance(result, Command)
        return result.update["tool_result_cache"]

    def test_cache_bounded_at_max_entries(self):
        axor = AxorMiddleware(cache_tools=["search"], max_tool_cache_entries=3)
        cache = {}
        for i in range(5):
            cache = self._drive(axor, "search", {"q": str(i)}, cache)
        assert len(cache) == 3

    def test_oldest_entries_evicted_first(self):
        axor = AxorMiddleware(cache_tools=["search"], max_tool_cache_entries=2)
        cache = {}
        for q in ("a", "b", "c"):  # 3 distinct misses, cap=2
            cache = self._drive(axor, "search", {"q": q}, cache)

        contents = set(cache.values())
        # "a" was inserted first → evicted; "b" and "c" remain.
        assert contents == {"OUT-{'q': 'b'}", "OUT-{'q': 'c'}"}

    def test_cache_hit_does_not_emit_state_update(self):
        # FIFO, not LRU: hits do not touch cache insertion order.
        axor = AxorMiddleware(cache_tools=["search"], max_tool_cache_entries=2)
        cache = {}
        cache = self._drive(axor, "search", {"q": "a"}, cache)
        cache = self._drive(axor, "search", {"q": "b"}, cache)

        # Re-hit "a" — should be a cache hit, returning ToolMessage.
        req = FakeToolCallRequest(
            tool_call={"name": "search", "args": {"q": "a"}, "id": "tc-a2"},
            state={"messages": [], "tool_result_cache": dict(cache)},
        )
        result = _drive_tool(axor, req, lambda r: pytest.fail("should not execute"))
        assert isinstance(result, ToolMessage)
        # No Command means LangGraph won't update state — `cache` from the
        # caller's POV is unchanged.

        # Insert a new key "c". With FIFO semantics the OLDEST insertion
        # ("a") is evicted, not the recently-accessed-but-stale "b".
        cache = self._drive(axor, "search", {"q": "c"}, cache)
        assert len(cache) == 2
        a_key = _hash_tool_call("search", {"q": "a"})
        b_key = _hash_tool_call("search", {"q": "b"})
        assert a_key not in cache  # FIFO: oldest evicted
        assert b_key in cache

    def test_unbounded_when_max_is_none(self):
        axor = AxorMiddleware(cache_tools=["search"], max_tool_cache_entries=None)
        cache = {}
        for i in range(50):
            cache = self._drive(axor, "search", {"q": str(i)}, cache)
        assert len(cache) == 50


# ── state_schema is wired ─────────────────────────────────────────────────────

class TestStateSchemaWired:

    def test_axor_middleware_declares_axor_state(self):
        """LangChain's create_agent reads state_schema from the middleware
        class to extend graph state. We need to verify it's set."""
        axor = AxorMiddleware()
        # The Impl class (returned by __new__) carries the attribute
        assert hasattr(axor, "state_schema")
        schema = axor.state_schema
        assert "tool_result_cache" in schema.__annotations__
