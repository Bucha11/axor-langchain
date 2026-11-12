"""What a LangChain agent shows the Control Plane.

The feeds themselves — trace, reputation, health, honesty — live in
:mod:`axor_wrap.observe`, shared with axor-claude. This module holds only
what is LangChain's: turning a ``ModelRequest`` into probe inputs, and probe
messages back into LangChain messages.
"""
from __future__ import annotations

from typing import Any

from axor_wrap.observe import (  # noqa: F401 - re-exported for callers and tests
    Experiment,
    HealthCheck,
    ReputationBook,
    RunRecorder,
    ToolRecord,
    _text,
    evidence_of,
    reconstruct_claim,
    taint_source,
)


def _to_lc(messages: list[dict]) -> list:
    """Provider-neutral probe messages → LangChain messages. A replayed tool
    result has no paired tool_call any more, so it is replayed as user content
    marked as a tool result — what the model read, without a dangling id."""
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    out = []
    for m in messages:
        role, content = m.get("role"), _text(m.get("content"))
        if role == "system":
            out.append(SystemMessage(content))
        elif role == "assistant":
            out.append(AIMessage(content))
        elif role == "tool":
            out.append(HumanMessage(f"[tool result]\n{content}"))
        else:
            out.append(HumanMessage(content))
    return out


def _raw(response: Any) -> dict:
    calls = [
        {"name": c.get("name"), "args": c.get("args", {})}
        for c in (getattr(response, "tool_calls", None) or [])
    ]
    return {"text": _text(getattr(response, "content", "")), "tool_calls": calls}


def _context_of(messages: list) -> tuple[list[dict], list[tuple[str, bool, dict]]]:
    """The run's last context as probe messages, plus localizer fragments:
    (fragment_id, pure_tainted, message). Tool results are the pure-tainted
    candidates and are keyed by tool_call_id — the ref an excision cuts by."""
    window, fragments = [], []
    for i, m in enumerate(messages):
        kind = getattr(m, "type", "")
        content = _text(getattr(m, "content", ""))
        if kind == "tool":
            msg = {"role": "tool", "content": content}
            fid = getattr(m, "tool_call_id", "") or f"tool_{i}"
            fragments.append((fid, True, msg))
        elif kind in ("ai", "assistant"):
            if not content:
                continue
            msg = {"role": "assistant", "content": content}
            fragments.append((getattr(m, "id", None) or f"msg_{i}", False, msg))
        elif kind == "system":
            continue
        else:
            msg = {"role": "user", "content": content}
            fragments.append((getattr(m, "id", None) or f"msg_{i}", False, msg))
        window.append(msg)
    return window, fragments



def health_inputs(request: Any) -> dict | None:
    """``HealthCheck.run`` arguments from the run's last ``ModelRequest``: the
    agent's own model (tools bound, so a tool-call attempt is observable), its
    system prompt, and the context as probe messages + localizer fragments."""
    model = getattr(request, "model", None)
    if model is None:
        return None
    tools = list(getattr(request, "tools", None) or [])
    bound = model.bind_tools(tools) if tools and hasattr(model, "bind_tools") else model
    system = getattr(request, "system_message", None)
    system_text = _text(getattr(system, "content", "")) if system is not None else ""

    async def infer(messages: list[dict]) -> dict:
        return _raw(await bound.ainvoke(_to_lc(messages)))

    def infer_sync(messages: list[dict]) -> dict:
        return _raw(bound.invoke(_to_lc(messages)))

    window, fragments = _context_of(list(getattr(request, "messages", []) or []))
    return {
        "infer": infer, "infer_sync": infer_sync, "system_text": system_text,
        "window": window, "fragments": fragments,
        "model": getattr(model, "model_name", "") or "",
    }
