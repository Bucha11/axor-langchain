from __future__ import annotations

from typing import Any

from axor_core.contracts.anomaly import NormalizedIntent
from axor_core.contracts.intent import Intent, IntentKind
from axor_core.errors.exceptions import NormalizerError, UnknownProviderFormatError
from axor_core.policy.normalizer import IntentNormalizer


class LangChainNormalizer:
    """
    ProviderNormalizer implementation for LangChain tool invocations.

    Converts LangChain tool invocation dicts to NormalizedIntent.

    Expected raw_event formats:
      - {"tool": str, "tool_input": dict}  — AgentAction / ToolCall form
      - {"name": str, "args": dict}         — newer ToolCall form

    Fail-closed: unknown formats raise UnknownProviderFormatError,
    missing tool name raises NormalizerError.
    """

    _PROVIDER = "langchain"

    def __init__(self, workdir: str | None = None) -> None:
        self._intent_normalizer = IntentNormalizer(workdir=workdir)

    def normalize(self, raw_event: Any) -> NormalizedIntent:
        tool_name, args = self._extract_tool_and_args(raw_event)
        intent = Intent(
            kind=IntentKind.TOOL_CALL,
            payload={"tool": tool_name, "args": args},
            node_id="",
        )
        return self._intent_normalizer.normalize(intent)

    def _extract_tool_and_args(self, raw_event: Any) -> tuple[str, dict]:
        if not isinstance(raw_event, dict):
            raise UnknownProviderFormatError(
                self._PROVIDER,
                type(raw_event).__name__,
            )
        # LangChain AgentAction / legacy form: {"tool": ..., "tool_input": ...}
        if "tool" in raw_event:
            tool_name = raw_event["tool"]
            args = raw_event.get("tool_input", raw_event.get("args", {}))
            if not isinstance(args, dict):
                raise NormalizerError(
                    self._PROVIDER,
                    f"tool_input must be a dict, got {type(args).__name__}",
                )
            return str(tool_name), args
        # Newer ToolCall form: {"name": ..., "args": ...}
        if "name" in raw_event:
            tool_name = raw_event["name"]
            args = raw_event.get("args", {})
            if not isinstance(args, dict):
                raise NormalizerError(
                    self._PROVIDER,
                    f"args must be a dict, got {type(args).__name__}",
                )
            return str(tool_name), args
        raise UnknownProviderFormatError(
            self._PROVIDER,
            f"dict missing 'tool' or 'name' key — got {list(raw_event.keys())}",
        )
