"""Control-Plane link for a LangChain agent.

LangChain owns the agent loop, so the node cannot hand its loop to axor-core.
What it can do is keep a connection of its own: :class:`PlaneLink` runs
axor-wrap's ``PlaneConnector`` (desired-state subscription + heartbeat) on a
private event loop in a daemon thread. That works the same under a sync
``agent.invoke`` and an async ``agent.ainvoke``, and it never borrows the
caller's loop.

Every mutation of the plane session — taking a queued excision or injection,
appending trace lines to the outbox — runs on the link's loop, so the SSE task
applying a desired-state delta and the agent thread consuming it never race.
Reads of posture (paused / stopped / cap) are plain attribute reads.

``AxorMiddleware`` builds one of these when given ``plane="https://..."`` or,
with no argument at all, when ``AXOR_PLANE_URL`` is set.
"""
from __future__ import annotations

import asyncio
import atexit
import logging
import os
import socket
import threading
from collections.abc import Callable
from typing import Any

_log = logging.getLogger("axor.langchain.plane")

ENV_URL = "AXOR_PLANE_URL"
ENV_NODE = "AXOR_NODE_ID"
ENV_KEY = "AXOR_INGEST_KEY"
ENV_TEST_BENCH = "AXOR_TEST_BENCH"

_INSTALL_HINT = "pip install 'axor-langchain[plane]'"


def default_node_id() -> str:
    return os.environ.get(ENV_NODE) or f"langchain-{socket.gethostname()}"


class PlaneLink:
    """A live node on the Control Plane, kept by a background thread.

    Exposes the surface :class:`AxorMiddleware` drives (``posture``,
    ``take_excision``, ``take_injection``, ``budget_cap_calls``) plus
    :meth:`emit` for the trace lines and reports the middleware produces.
    """

    def __init__(
        self,
        url: str,
        node_id: str | None = None,
        *,
        ingest_key: str | None = None,
        operator_keys: dict[str, str] | None = None,
        test_bench: bool | None = None,
        local_budget_cap: int | None = None,
        heartbeat_period: float = 10.0,
        start: bool = True,
    ) -> None:
        try:
            from axor_wrap.connect import PlaneConnector
            from axor_wrap.plane.admission import PlaneAdmission
        except ImportError as exc:
            raise RuntimeError(
                f"connecting to the Control Plane needs axor-wrap[plane] — {_INSTALL_HINT}"
            ) from exc
        if test_bench is None:
            test_bench = os.environ.get(ENV_TEST_BENCH, "") not in ("", "0", "false")
        self.url = url.rstrip("/")
        self.node_id = node_id or default_node_id()
        self.connector = PlaneConnector(
            self.url, self.node_id,
            ingest_key=ingest_key or os.environ.get(ENV_KEY) or None,
            operator_keys=operator_keys,
            test_bench=test_bench,
            local_budget_cap=local_budget_cap,
            heartbeat_period=heartbeat_period,
        ).connect()
        self.session = self.connector.session
        self._admission = PlaneAdmission(self.session)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._closed = False
        if start:
            self.start()

    @classmethod
    def from_env(cls, **kw: Any) -> PlaneLink | None:
        url = os.environ.get(ENV_URL)
        return cls(url, **kw) if url else None

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> PlaneLink:
        if self._thread is not None:
            return self
        self._thread = threading.Thread(
            target=self._run, name=f"axor-plane-{self.node_id}", daemon=True,
        )
        self._thread.start()
        self._ready.wait(5.0)
        atexit.register(self.close)
        return self

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        self._ready.set()
        try:
            loop.run_until_complete(self.connector.run())
        except Exception:  # noqa: BLE001 - a dead link must not kill the agent
            _log.exception("axor: Control-Plane link stopped")
        finally:
            loop.close()

    def close(self, timeout: float = 5.0) -> None:
        """Deliver what is queued (operator acks, trace lines), then stop."""
        if self._closed:
            return
        self._closed = True
        loop = self._loop
        if loop is None or not loop.is_running():
            return
        try:
            asyncio.run_coroutine_threadsafe(
                self.connector.flush(), loop,
            ).result(timeout)
        except Exception:  # noqa: BLE001 - best effort on the way out
            _log.debug("axor: final flush failed", exc_info=True)
        loop.call_soon_threadsafe(self.connector.stop)
        if self._thread is not None:
            self._thread.join(timeout)

    # ── on the link's loop ────────────────────────────────────────────────────

    def _on_loop(self, fn: Callable[[], Any], timeout: float = 5.0) -> Any:
        loop = self._loop
        if loop is None or not loop.is_running() or threading.current_thread() is self._thread:
            return fn()

        async def call() -> Any:
            return fn()

        return asyncio.run_coroutine_threadsafe(call(), loop).result(timeout)

    def submit(self, coro_fn: Callable[[], Any]) -> None:
        """Run an async callable on the link's loop, fire-and-forget."""
        loop = self._loop
        if loop is None or not loop.is_running():
            return
        fut = asyncio.run_coroutine_threadsafe(coro_fn(), loop)
        fut.add_done_callback(_log_failure)

    def emit(self, lines: list[dict]) -> None:
        """Queue kernel trace lines; they ride the next flush with a heartbeat."""
        if not lines:
            return
        self.submit(lambda: self.connector.flush(list(lines)))

    # ── the surface AxorMiddleware drives ─────────────────────────────────────

    def posture(self) -> str:
        return self._admission.posture()

    def budget_cap_calls(self) -> int | None:
        return self._admission.budget_cap_calls()

    def take_excision(self, provenance):  # noqa: ANN001, ANN201 - core types
        return self._on_loop(lambda: self._admission.take_excision(provenance))

    def take_injection(self):  # noqa: ANN201 - core type
        return self._on_loop(self._admission.take_injection)


def _log_failure(fut: Any) -> None:
    if fut.cancelled():
        return
    exc = fut.exception()
    if exc is not None:
        _log.warning("axor: Control-Plane delivery failed: %s", exc)
