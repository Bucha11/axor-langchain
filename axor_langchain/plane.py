"""Control-Plane link for a LangChain agent.

LangChain owns the agent loop, so the node keeps a connection of its own:
axor-wrap's :class:`~axor_wrap.plane.link.PlaneLink`, a background thread
with a private event loop, the same under ``agent.invoke`` and
``agent.ainvoke``. ``AxorMiddleware`` builds one when given
``plane="https://..."`` or, with no argument at all, when ``AXOR_PLANE_URL``
is set.
"""
from __future__ import annotations

import os
import socket
from typing import Any

ENV_URL = "AXOR_PLANE_URL"
ENV_NODE = "AXOR_NODE_ID"
ENV_KEY = "AXOR_INGEST_KEY"
ENV_TEST_BENCH = "AXOR_TEST_BENCH"


def default_node_id() -> str:
    return os.environ.get(ENV_NODE) or f"langchain-{socket.gethostname()}"


def PlaneLink(url: str, node_id: str | None = None, **kw: Any):  # noqa: N802 - a factory
    """Build axor-wrap's PlaneLink for this agent (node ids default to
    ``langchain-<host>``). Raises RuntimeError without the plane extra."""
    try:
        from axor_wrap.plane.link import PlaneLink as _Link
    except ImportError as exc:  # pragma: no cover - wrap is a base dependency
        raise RuntimeError(
            "connecting to the Control Plane needs axor-wrap[plane] — "
            "pip install 'axor-langchain[plane]'"
        ) from exc
    kw.setdefault("node_prefix", "langchain")
    return _Link(url, node_id, **kw)
