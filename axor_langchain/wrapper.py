from __future__ import annotations

import logging
import inspect
from typing import Any, Callable

from pydantic import PrivateAttr

from axor_langchain.normalizer import LangChainNormalizer

_log = logging.getLogger("axor.langchain.wrapper")

_OBSERVABILITY_ONLY_WARNING = (
    "AxorToolWrapper provides observability only — "
    "pass policy_fn for enforcement"
)


def _make_tool_class():
    from langchain_core.tools import BaseTool

    class AxorToolWrapper(BaseTool):
        """
        Governance wrapper around a LangChain BaseTool.

        Intercepts _run()/_arun() and evaluates policy before the original tool
        receives control. If denied, returns a structured denial string without
        ever calling the underlying tool.

        The wrapped tool is not accessible via the public API.

        policy_fn: Callable[[NormalizedIntent], str | None]
            Called before each tool execution. Return None to allow, or a
            denial reason string to block.

        observability_callback: optional — called after policy evaluation with
            (tool_name, normalized_intent, denial_reason | None). Does not
            affect enforcement.
        """

        _wrapped: Any = PrivateAttr()
        _policy_fn: Any = PrivateAttr()
        _normalizer: LangChainNormalizer = PrivateAttr()
        _obs_callback: Any = PrivateAttr()

        def __init__(
            self,
            tool: BaseTool,
            policy_fn: Callable | None = None,
            observability_callback: Callable | None = None,
            **kwargs: Any,
        ) -> None:
            super().__init__(
                name=tool.name,
                description=tool.description or "",
                **kwargs,
            )
            self._wrapped = tool
            self._policy_fn = policy_fn
            self._normalizer = LangChainNormalizer()
            self._obs_callback = observability_callback

            if policy_fn is None:
                _log.warning(_OBSERVABILITY_ONLY_WARNING)
            elif inspect.iscoroutinefunction(policy_fn):
                raise TypeError("AxorToolWrapper policy_fn must be synchronous")

        def _evaluate(self, tool_input: dict) -> str | None:
            if self._policy_fn is None:
                return None
            try:
                normalized = self._normalizer.normalize(
                    {"tool": self.name, "args": tool_input}
                )
            except Exception:
                _log.error(
                    "normalizer raised for tool '%s' — failing closed",
                    self.name, exc_info=True,
                )
                return "normalizer_error: tool blocked due to normalization failure"
            denial = self._policy_fn(normalized)
            if self._obs_callback is not None:
                try:
                    self._obs_callback(self.name, normalized, denial)
                except Exception:
                    pass
            return denial

        def _run(self, *args: Any, **kwargs: Any) -> str:
            tool_input = kwargs if kwargs else ({"input": args[0]} if args else {})
            denial = self._evaluate(tool_input)
            if denial:
                return f"tool_denied: {denial}"
            return self._wrapped._run(*args, **kwargs)

        async def _arun(self, *args: Any, **kwargs: Any) -> str:
            tool_input = kwargs if kwargs else ({"input": args[0]} if args else {})
            denial = self._evaluate(tool_input)
            if denial:
                return f"tool_denied: {denial}"
            return await self._wrapped._arun(*args, **kwargs)

    return AxorToolWrapper


_AxorToolWrapperClass: type | None = None


def _get_wrapper_class() -> type:
    global _AxorToolWrapperClass
    if _AxorToolWrapperClass is None:
        _AxorToolWrapperClass = _make_tool_class()
    return _AxorToolWrapperClass


class AxorToolWrapper:
    """
    Lazy facade that constructs the real Pydantic-based wrapper only when
    langchain_core is available. Import-time safe.
    """

    def __new__(cls, tool: Any, **kwargs: Any) -> Any:
        return _get_wrapper_class()(tool, **kwargs)
