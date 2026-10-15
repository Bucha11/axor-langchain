"""The kernel gate on LangChain tool calls.

``AxorMiddleware.wrap_tool_call`` and ``AxorToolWrapper`` both run a call through one
``ToolGate``: the user's request seeds it (trusted origin + L1 grants), every call is
decided by the current kernel (``gate.Gate`` around a ``ToolCallGovernor``) before the
tool runs, the tool runs on the arguments the kernel resolved, and its output is
registered and returned as the agent may see it (references for hidden values).
"""
from __future__ import annotations

from typing import Any

from axor_langchain.gate import Admitted, Gate, Refused


def _human_texts(messages: Any) -> list[str]:
    out = []
    for m in messages or []:
        if getattr(m, "type", None) != "human":
            continue
        c = getattr(m, "content", "")
        if isinstance(c, list):
            c = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in c)
        if isinstance(c, str) and c:
            out.append(c)
    return out


class ToolGate:
    """One agent session's gate. Share one between the middleware and wrapped tools."""

    def __init__(self, gate: Gate) -> None:
        self.gate = gate
        self._requests: set[str] = set()

    @classmethod
    def from_config(cls, governance: Any = None, *, confirmer: Any = None,
                    structured: Any = None) -> "ToolGate":
        return cls(Gate.from_config(governance or {}, confirmer=confirmer, structured=structured))

    def request(self, text: str) -> None:
        """The user's own request (each new one once)."""
        if text and text not in self._requests:
            self._requests.add(text)
            self.gate.begin(text)

    def observe_state(self, state: Any) -> None:
        """Seed from the human messages of the agent state."""
        msgs = state.get("messages") if isinstance(state, dict) else getattr(state, "messages", None)
        for text in _human_texts(msgs):
            self.request(text)

    def check(self, tool: str, args: dict, call_key: str | None = None) -> "Admitted | Refused":
        return self.gate.check(tool, args or {}, call_key=call_key)

    def finish(self, adm: Admitted, tool: str, output: Any) -> Any:
        return self.gate.finish(adm, tool, output)

    def fail(self, adm: Admitted) -> None:
        self.gate.fail(adm)
