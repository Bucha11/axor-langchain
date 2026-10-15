"""One governed tool call on the current axor kernel (``ToolCallGovernor``).

The protocol is the daemon's (``axor_daemon/enforcer.py``) and the one the AgentDojo
shim replays (axor-eval ``experiments/ajar/axor_ajar.py``), in the same order:

    begin(task)           the user's request: trusted origin + L1 grants from it
    check(tool, args)     evaluate -> refuse, or the arguments the tool must run with
                          (references resolved, MODEL.md 9) and a reserved grant use
    finish(adm, output)   commit the use, register the output, return what the agent
                          sees (references for hidden values); a declared description
                          source's structure goes to the gate (MODEL.md 9.15)
    fail(adm)             the tool raised: the use stays spent, nothing is registered

``lock`` makes check (evaluate + reserve) and finish (commit + register) atomic; an
adapter whose framework runs tool calls concurrently holds it (or an asyncio lock of its
own) from check to finish, so a call is decided on the session's state at the moment it
runs, as the daemon's per-session lock does.

Nothing here decides: every verdict is the governor's. This file is byte-identical in
axor-claude, axor-openrouter, axor-langchain and axor-wrap; the compatibility check
compares the four digests, so change it in all of them or in none.
"""
from __future__ import annotations

import copy
import os
import threading
from dataclasses import dataclass
from typing import Any, Callable

GATE_PROTOCOL = "axor-gate/1"


@dataclass
class Admitted:
    decision: Any
    args: dict[str, Any]  # what the tool runs with: the kernel's resolved arguments
    spend: str | None  # the reserved grant use (committed by finish / fail)


@dataclass
class Refused:
    reason: str
    category: str
    decision: Any = None

    def message(self) -> str:
        """The refusal as the agent reads it."""
        return f"tool_denied: governance ({self.category}): {self.reason}"


class Gate:
    """A ``ToolCallGovernor`` and the per-call protocol around it. One per agent session."""

    def __init__(self, governor: Any, *,
                 structured: Callable[[str, Any], Any] | None = None) -> None:
        self.governor = governor
        self._structured = structured
        self.lock = threading.RLock()

    @classmethod
    def from_config(cls, config: Any, *, confirmer: Any = None,
                    structured: Callable[[str, Any], Any] | None = None,
                    **overrides: Any) -> "Gate":
        """``config``: a ``GovernanceConfig``, the path of its YAML, or its mapping.
        ``confirmer`` is the operator's L4 channel; without one nothing is confirmed."""
        from axor_core.config import GovernanceConfig
        from axor_core.governor import ToolCallGovernor

        if isinstance(config, (str, os.PathLike)):
            config = GovernanceConfig.from_yaml(os.fspath(config))
        elif isinstance(config, dict):
            config = GovernanceConfig.from_dict(config)
        kw = config.as_governor_kwargs()
        if confirmer is not None:
            kw["confirmer"] = confirmer
        kw.update(overrides)
        return cls(ToolCallGovernor(**kw), structured=structured)

    def begin(self, task: str) -> None:
        """The user's own request (never model-written text)."""
        with self.lock:
            g = self.governor
            g.register_task(task)
            if getattr(g, "_grant_args", None):
                g.grants.from_request(task)

    def check(self, tool: str, args: dict[str, Any],
              call_key: str | None = None) -> "Admitted | Refused":
        with self.lock:
            g = self.governor
            d = g.evaluate(tool, dict(args or {}))
            if not d.allowed:
                return Refused(d.reason, d.category, d)
            resolved = getattr(d, "resolved_args", None)
            run = copy.deepcopy(resolved if resolved is not None else dict(args or {}))
            key = g.spend(d, run, call_key=call_key)
            if getattr(d, "grant_id", None) is not None and key is None:
                return Refused("the grant was used up concurrently", "grant_gate", d)
            return Admitted(d, run, key)

    def finish(self, adm: Admitted, tool: str, output: Any) -> Any:
        with self.lock:
            return self._finish(adm, tool, output)

    def _finish(self, adm: Admitted, tool: str, output: Any) -> Any:
        g = self.governor
        if adm.spend is not None:
            g.grant_store.commit(adm.spend)
        text = output if isinstance(output, str) else repr(output)
        if getattr(g, "_xrefs", None) is not None:
            view = g.present_output(adm.decision, text)
        else:
            g.register_output(adm.decision, text)
            view = output
        if getattr(g, "_delegated", None):
            g.observe_delegated(tool, adm.args, text)  # MODEL.md 9.16, off unless configured
        desc = getattr(g, "_desc", None)
        if self._structured is not None and desc is not None and tool in desc.sources:
            value = self._structured(tool, output)
            if value is not None:
                g.observe_structured(tool, adm.args, value)
        return view

    def fail(self, adm: Admitted) -> None:
        with self.lock:
            if adm.spend is not None:
                self.governor.grant_store.commit(adm.spend)
