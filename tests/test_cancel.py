"""Cancellation surface for AxorMiddleware.

Previously the middleware declared a `_cancelled` flag that nothing read,
so an external `cancel()` had no effect. These tests pin down the new
external API:

  - `AxorMiddleware.cancel()` flips the flag (idempotent, safe from any thread).
  - The next `wrap_model_call` / `wrap_tool_call` invocation raises
    `AxorCancelledError` instead of dispatching to the underlying handler.
"""
from __future__ import annotations

import asyncio

import pytest

from axor_langchain import AxorCancelledError, AxorMiddleware


def _make_middleware():
    # Construct a middleware with the smallest possible config so we don't
    # exercise compression / classification / telemetry side effects.
    return AxorMiddleware(
        soft_token_limit=10_000,
        verbose=False,
        telemetry="off",
    )


class _FakeModelRequest:
    """Minimum surface that _prepare_model_request reads from."""

    def __init__(self):
        self.messages = []
        self.tools = []
        self.system_message = None
        self.model = "test"
        self.runnable_config = None
        self.state = {}

    def override(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        return self


def test_cancel_is_idempotent():
    mw = _make_middleware()
    assert mw.is_cancelled() is False
    mw.cancel("first")
    assert mw.is_cancelled() is True
    # Second call must not flip back or change reason.
    mw.cancel("second")
    assert mw.is_cancelled() is True


def test_wrap_model_call_raises_after_cancel():
    mw = _make_middleware()
    mw.cancel()

    handler_called = []

    def handler(req):
        handler_called.append(req)
        return None

    with pytest.raises(AxorCancelledError):
        mw.wrap_model_call(_FakeModelRequest(), handler)

    assert handler_called == [], "handler must not run after cancel()"


def test_awrap_model_call_raises_after_cancel():
    mw = _make_middleware()
    mw.cancel()

    handler_called = []

    async def handler(req):
        handler_called.append(req)
        return None

    async def go():
        with pytest.raises(AxorCancelledError):
            await mw.awrap_model_call(_FakeModelRequest(), handler)

    asyncio.run(go())
    assert handler_called == []


def test_reset_clears_cancel_flag():
    mw = _make_middleware()
    mw.cancel()
    assert mw.is_cancelled()
    mw.reset()
    assert not mw.is_cancelled()
