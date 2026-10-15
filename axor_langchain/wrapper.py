from __future__ import annotations

import logging
import inspect
from typing import Any, Callable

from pydantic import PrivateAttr

from axor_langchain.normalizer import LangChainNormalizer

_log = logging.getLogger("axor.langchain.wrapper")

_OWN_GATE_WARNING = (
    "AxorToolWrapper without a shared gate: this tool is gated under an empty "
    "configuration with a ledger of its own — pass gate= (AxorMiddleware.wrap_tools does)"
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

        gate: the agent session's ``ToolGate`` (``AxorMiddleware.wrap_tools``
            passes its own). After ``policy_fn`` every call is decided by the
            current kernel, the tool runs on the arguments it resolved and its
            output is registered; without one the wrapper gates under an empty
            configuration with a ledger of its own (no request: call
            ``wrapper._gate.request(task)``).
        """

        _wrapped: Any = PrivateAttr()
        _policy_fn: Any = PrivateAttr()
        _normalizer: LangChainNormalizer = PrivateAttr()
        _obs_callback: Any = PrivateAttr()
        _gate: Any = PrivateAttr()

        def __init__(
            self,
            tool: BaseTool,
            policy_fn: Callable | None = None,
            observability_callback: Callable | None = None,
            gate: Any = None,
            **kwargs: Any,
        ) -> None:
            if getattr(tool, "args_schema", None) is not None:
                kwargs.setdefault("args_schema", tool.args_schema)
            super().__init__(
                name=tool.name,
                description=tool.description or "",
                **kwargs,
            )
            self._wrapped = tool
            self._policy_fn = policy_fn
            self._normalizer = LangChainNormalizer()
            self._obs_callback = observability_callback
            if gate is None:
                from axor_langchain.governed import ToolGate
                _log.warning(_OWN_GATE_WARNING)
                gate = ToolGate.from_config()
            self._gate = gate

            if policy_fn is not None and inspect.iscoroutinefunction(policy_fn):
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

        def _admit(self, args: tuple, kwargs: dict):
            """policy_fn, then the kernel gate: (Admitted, args, kwargs) or a refusal."""
            from axor_langchain.governed import Admitted
            tool_input = kwargs if kwargs else ({"input": args[0]} if args else {})
            denial = self._evaluate(tool_input)
            if denial:
                return f"tool_denied: {denial}"
            verdict = self._gate.check(self.name, tool_input)
            if not isinstance(verdict, Admitted):
                return verdict.message()
            if verdict.args != tool_input:  # MODEL.md 9: resolved references
                if kwargs or not args:
                    kwargs = dict(verdict.args)
                else:
                    args = (verdict.args.get("input", args[0]),) + tuple(args[1:])
            return verdict, args, kwargs

        def _run(self, *args: Any, **kwargs: Any) -> str:
            with self._gate.gate.lock:  # one call at a time, check to registration
                return self._run_locked(args, kwargs)

        async def _arun(self, *args: Any, **kwargs: Any) -> str:
            async with self._gate.aserial:
                return await self._arun_locked(args, kwargs)

        def _run_locked(self, args: tuple, kwargs: dict) -> str:
            admitted = self._admit(args, kwargs)
            if isinstance(admitted, str):
                return admitted
            verdict, args, kwargs = admitted
            try:
                out = self._wrapped.invoke(kwargs if kwargs or not args else args[0])
            except BaseException:
                self._gate.fail(verdict)
                raise
            return self._gate.finish(verdict, self.name, out)

        async def _arun_locked(self, args: tuple, kwargs: dict) -> str:
            admitted = self._admit(args, kwargs)
            if isinstance(admitted, str):
                return admitted
            verdict, args, kwargs = admitted
            try:
                out = await self._wrapped.ainvoke(kwargs if kwargs or not args else args[0])
            except BaseException:
                self._gate.fail(verdict)
                raise
            return self._gate.finish(verdict, self.name, out)

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
