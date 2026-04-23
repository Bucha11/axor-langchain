from __future__ import annotations

"""
axor-langchain — governance middleware for LangChain 1.0 agents.

Uses axor-core engines for real context compression and budget tracking:
  - ContextManager + ContextCompressor  → fragment-aware compression
  - BudgetTracker + BudgetPolicyEngine  → per-turn optimization decisions
  - ToolPolicy                          → tool allow/deny governance

The actual AgentMiddleware subclass is built lazily so the package is
importable without langchain installed (useful for unit tests).
"""

import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


_TELEMETRY_MARKER = Path.home() / ".axor" / ".telemetry_notice_shown"


def _maybe_show_telemetry_notice() -> None:
    """
    Print a single-line stderr notice about anonymous telemetry the first
    time AxorMiddleware is constructed on a given machine. Idempotent.
    Suppressed by AXOR_NO_BANNER=1.
    """
    if os.environ.get("AXOR_NO_BANNER") == "1":
        return
    if _TELEMETRY_MARKER.is_file():
        return
    sys.stderr.write(
        "axor: anonymous telemetry is off. "
        "Set AXOR_TELEMETRY=local or pass telemetry='local' to help tune "
        "the classifier. (shown once; suppress with AXOR_NO_BANNER=1)\n"
    )
    try:
        _TELEMETRY_MARKER.parent.mkdir(parents=True, exist_ok=True)
        _TELEMETRY_MARKER.write_text("shown\n", encoding="utf-8")
    except OSError:
        return


def _resolve_telemetry_mode(kwarg_value: str | None) -> str:
    """kwarg wins; then AXOR_TELEMETRY env; else 'off'."""
    if kwarg_value is not None:
        return str(kwarg_value).lower()
    env = os.environ.get("AXOR_TELEMETRY")
    if env:
        return env.lower()
    return "off"


def _build_telemetry_pipeline(mode: str, axor_version: str) -> Any | None:
    """
    Construct a pipeline for the given resolved mode string. Returns None
    when axor-telemetry is not installed or mode is 'off'. Never raises.
    """
    if mode == "off":
        return None
    try:
        from axor_telemetry import TelemetryConfig, TelemetryMode, build_pipeline
    except ImportError:
        return None
    try:
        tmode = TelemetryMode(mode)
    except ValueError:
        return None
    base = TelemetryConfig.load()
    cfg = TelemetryConfig(
        mode=tmode,
        endpoint=base.endpoint,
        queue_path=base.queue_path,
        fingerprint_kind=base.fingerprint_kind,
    )
    try:
        return build_pipeline(config=cfg, axor_version=axor_version)
    except Exception:
        return None


# ── Helpers ────────────────────────────────────────────────────────────────────

def _msg_text(msg) -> str:
    """Extract plain text from a LangChain message."""
    content = getattr(msg, "content", "") or ""
    if isinstance(content, list):
        return " ".join(
            b.get("text", "") for b in content if isinstance(b, dict)
        )
    return str(content)


def _msg_tokens(msg) -> int:
    return max(len(_msg_text(msg)) // 4, 1)


def _tool_name(tool_call) -> str:
    if isinstance(tool_call, dict):
        return tool_call.get("name", "unknown")
    return getattr(tool_call, "name", "unknown")


def _tool_call_id(tool_call) -> str:
    if isinstance(tool_call, dict):
        return tool_call.get("id", "")
    return getattr(tool_call, "id", "")


# ── Per-tool stats ─────────────────────────────────────────────────────────────

@dataclass
class ToolCallStats:
    call_count: int = 0
    error_count: int = 0
    total_latency_ms: float = 0.0
    denied_count: int = 0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.call_count if self.call_count else 0.0

    @property
    def success_rate(self) -> float:
        return (self.call_count - self.error_count) / self.call_count if self.call_count else 1.0

    @property
    def effective_rate(self) -> float:
        """Success rate accounting for both errors and denials."""
        total = self.call_count + self.denied_count
        return (self.call_count - self.error_count) / total if total else 1.0


@dataclass
class ToolStats:
    by_tool: dict[str, ToolCallStats] = field(default_factory=dict)

    def _get(self, name: str) -> ToolCallStats:
        if name not in self.by_tool:
            self.by_tool[name] = ToolCallStats()
        return self.by_tool[name]

    def record_call(self, name: str, latency_ms: float) -> None:
        s = self._get(name)
        s.call_count += 1
        s.total_latency_ms += latency_ms

    def record_error(self, name: str) -> None:
        self._get(name).error_count += 1

    def record_denied(self, name: str) -> None:
        self._get(name).denied_count += 1

    @property
    def total_calls(self) -> int:
        return sum(s.call_count for s in self.by_tool.values())

    @property
    def total_errors(self) -> int:
        return sum(s.error_count for s in self.by_tool.values())

    def summary(self) -> str:
        lines = [
            f"Tool calls: {self.total_calls} total, "
            f"{self.total_errors} errors, "
            f"{sum(s.denied_count for s in self.by_tool.values())} denied"
        ]
        for name, s in sorted(self.by_tool.items()):
            lines.append(
                f"  {name}: {s.call_count} calls, "
                f"{s.avg_latency_ms:.0f}ms avg, "
                f"{s.success_rate:.0%} success"
            )
        return "\n".join(lines)

    def reset(self) -> None:
        self.by_tool.clear()


# ── axor-core bridge ───────────────────────────────────────────────────────────

def _make_lineage(node_id: str):
    """Minimal LineageSummary for root-level LangChain agents."""
    from axor_core.contracts.context import LineageSummary
    return LineageSummary(
        node_id=node_id,
        parent_id=None,
        depth=0,
        ancestry_ids=[],
        inherited_restrictions=[],
    )


def _make_raw_state(task: str, session_id: str, fragments_text: list[str]):
    """Wrap LangChain messages into axor-core RawExecutionState."""
    from axor_core.contracts.context import RawExecutionState
    return RawExecutionState(
        task=task,
        session_id=session_id,
        parent_export=None,
        session_state={},
        memory_fragments=fragments_text,
        lineage=None,
    )


def _messages_to_fragments(messages: list):
    """
    Convert LangChain messages to axor-core ContextFragments.

    Message type mapping:
      system   → kind="fact",         value="pinned"
      human    → kind="fact",         value="working"
      ai       → kind="assistant_prose", value="working"
      tool     → kind="tool_result",  value="working"  (or "ephemeral" if old)
    """
    from axor_core.contracts.context import ContextFragment

    fragments = []
    for i, msg in enumerate(messages):
        mtype   = getattr(msg, "type", "")
        content = _msg_text(msg)
        if not content:
            continue
        tokens  = max(len(content) // 4, 1)
        # age-based value: older tool results become ephemeral
        age     = len(messages) - i

        if mtype == "system":
            kind, value = "fact", "pinned"
        elif mtype == "human":
            kind, value = "fact", "working"
        elif mtype == "ai":
            kind, value = "assistant_prose", "working"
        elif mtype == "tool":
            kind  = "tool_result"
            value = "ephemeral" if age > 4 else "working"
        else:
            kind, value = "fact", "working"

        fragments.append(ContextFragment(
            kind=kind,
            content=content,
            token_estimate=tokens,
            source=f"langchain:{mtype}:{i}",
            relevance=1.0 if age <= 2 else max(0.3, 1.0 - age * 0.08),
            value=value,
            turn=0,  # turn tracking managed by middleware, not core
        ))
    return fragments


def _fragments_to_messages(
    context_view,
    original_messages: list,
    personality: str | None,
):
    """
    Reconstruct a LangChain message list from a ContextView.

    Strategy:
      1. Always keep original system messages (they are pinned in axor anyway).
      2. Always keep the last human message verbatim.
      3. Replace the rest with compressed fragments from ContextView.
      4. Re-attach tool-call AI messages paired with their ToolMessages
         so the conversation remains structurally valid for the model API.

    Returns (new_messages, was_changed).
    """
    try:
        from langchain_core.messages import SystemMessage
    except ImportError:
        # benchmark / test context — use a simple stub
        class SystemMessage:  # type: ignore[no-redef]
            def __init__(self, content: str):
                self.type    = "system"
                self.content = content

    system_msgs = [m for m in original_messages if getattr(m, "type", None) == "system"]
    last_human  = next(
        (m for m in reversed(original_messages) if getattr(m, "type", None) == "human"),
        None,
    )

    # Build middle part from compressed fragments
    # Skip pinned (system) and pure prose fragments — they come from system/last_human
    middle: list = []
    for frag in context_view.visible_fragments:
        if frag.value == "pinned":
            continue  # already covered by system_msgs
        if frag.kind == "tool_result":
            # Try to find original ToolMessage for this fragment
            src_idx = _parse_source_index(frag.source)
            if src_idx is not None and src_idx < len(original_messages):
                orig = original_messages[src_idx]
                if frag.content != _msg_text(orig):
                    # Content was compressed — create a copy with truncated content
                    try:
                        orig = orig.model_copy(update={"content": frag.content})
                    except AttributeError:
                        import copy
                        orig = copy.copy(orig)
                        orig.content = frag.content
                middle.append(orig)
            # else: fragment came from compressor summary — skip (no paired AI msg)
        elif frag.kind == "assistant_prose":
            src_idx = _parse_source_index(frag.source)
            if src_idx is not None and src_idx < len(original_messages):
                middle.append(original_messages[src_idx])

    # Ensure every ToolMessage has its paired AI message (tool_calls present)
    middle = _repair_tool_pairs(middle, original_messages)

    # Assemble: system → personality → middle → last_human
    result = list(system_msgs)
    if personality and not system_msgs:
        result.append(SystemMessage(content=personality))
    elif personality and system_msgs:
        merged = f"{personality}\n\n{_msg_text(system_msgs[0])}"
        try:
            result[0] = system_msgs[0].model_copy(update={"content": merged})
        except AttributeError:
            import copy
            try:
                result[0] = copy.copy(system_msgs[0])
                result[0].content = merged
            except (AttributeError, TypeError):
                pass  # truly immutable — keep original

    result.extend(m for m in middle if getattr(m, "type", None) not in ("system", "human"))
    if last_human is not None:
        result.append(last_human)

    return result, result != original_messages


def _parse_source_index(source: str) -> int | None:
    """Extract the original message index from a fragment source string."""
    try:
        return int(source.split(":")[-1])
    except (ValueError, IndexError):
        return None


def _repair_tool_pairs(middle: list, originals: list) -> list:
    """
    Ensure every ToolMessage in `middle` is preceded by the AI message
    that issued the tool call. Without this, most model APIs reject the input.
    """
    # build index: tool_call_id → AI message that issued it
    tool_call_to_ai: dict[str, Any] = {}
    for msg in originals:
        if getattr(msg, "type", None) == "ai":
            for tc in getattr(msg, "tool_calls", []) or []:
                tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                if tc_id:
                    tool_call_to_ai[tc_id] = msg

    repaired = []
    seen_ai: set[int] = set()
    for msg in middle:
        if getattr(msg, "type", None) == "tool":
            tc_id = getattr(msg, "tool_call_id", None)
            ai_msg = tool_call_to_ai.get(tc_id) if tc_id else None
            if ai_msg is not None and id(ai_msg) not in seen_ai:
                repaired.append(ai_msg)
                seen_ai.add(id(ai_msg))
        repaired.append(msg)
    return repaired


# ── Core governance (no LangChain imports) ─────────────────────────────────────

class _AxorGovernanceCore:
    """
    Pure governance logic — no LangChain imports.

    Uses axor-core engines:
      ContextManager      → fragment-aware compression via ContextCompressor
      BudgetTracker       → tracks token spend per turn
      BudgetPolicyEngine  → decides when to compress harder or stop
    """

    # Below this token count, compression pipeline is skipped entirely.
    # Governance (tool filtering, budget tracking) still applies.
    _BYPASS_TOKEN_THRESHOLD = 4000

    def __init__(
        self,
        soft_token_limit: int | None = None,
        hard_token_limit: int | None = None,
        compression_mode: str = "auto",
        bypass_token_threshold: int | None = None,
        allowed_tools: list[str] | None = None,
        denied_tools:  list[str] | None = None,
        personality:   str | None = None,
        memory_provider=None,
        memory_namespace: str = "axor",
        tool_error_handler: Callable[[str, Exception], str] | None = None,
        tool_max_retries: int = 0,
        tool_retry_delay: float = 0.0,
        track_tool_stats: bool = False,
        verbose: bool = False,
        telemetry: str | None = None,
    ) -> None:
        self._soft_limit   = soft_token_limit
        self._hard_limit   = hard_token_limit or (
            int(soft_token_limit * 1.5) if soft_token_limit else None
        )
        self._compression_mode = compression_mode
        self._bypass_threshold = bypass_token_threshold if bypass_token_threshold is not None else self._BYPASS_TOKEN_THRESHOLD
        self._allowed_tools    = set(allowed_tools) if allowed_tools else None
        self._denied_tools     = set(denied_tools)  if denied_tools  else set()
        self._personality      = personality
        self._memory_provider  = memory_provider
        self._memory_namespace = memory_namespace
        self._tool_error_handler = tool_error_handler
        self._tool_max_retries   = tool_max_retries
        self._tool_retry_delay   = tool_retry_delay
        self._track_tool_stats   = track_tool_stats
        self._verbose            = verbose

        self._turn        = 0
        self._cancelled   = False
        self._session_id  = f"lc-{id(self):x}"
        self._tool_stats  = ToolStats() if track_tool_stats else None

        # axor-core engines
        self._ctx_manager: Any = None
        self._budget_tracker: Any = None
        self._budget_engine: Any = None
        self._engines_ready = False

        # Telemetry: build a pipeline when caller opts in via kwarg or
        # AXOR_TELEMETRY env. A one-time notice is printed to stderr when
        # telemetry is off, so new users discover the option.
        self._telemetry_mode = _resolve_telemetry_mode(telemetry)
        self._telemetry = _build_telemetry_pipeline(
            mode=self._telemetry_mode,
            axor_version=self._axor_version(),
        )
        if self._telemetry_mode == "off":
            _maybe_show_telemetry_notice()

    @staticmethod
    def _axor_version() -> str:
        try:
            from axor_langchain import __version__  # type: ignore
            return __version__
        except Exception:
            return ""

    # ── lazy engine init ──────────────────────────────────────────────────────

    def _ensure_engines(self) -> bool:
        """Init axor-core engines on first use. Returns False if unavailable."""
        if self._engines_ready:
            return True
        try:
            from axor_core.context.manager import ContextManager
            from axor_core.budget.tracker import BudgetTracker
            from axor_core.budget.estimator import BudgetEstimator
            from axor_core.budget.policy_engine import BudgetPolicyEngine

            self._ctx_manager     = ContextManager()
            self._budget_tracker  = BudgetTracker()
            estimator             = BudgetEstimator()
            self._budget_engine   = BudgetPolicyEngine(
                tracker=self._budget_tracker,
                estimator=estimator,
                soft_limit=self._soft_limit,
            )
            self._budget_tracker.register_node(
                node_id=self._session_id,
                parent_id=None,
                depth=0,
            )
            self._engines_ready = True
            return True
        except ImportError:
            if self._verbose:
                print("[axor] axor-core not installed — using built-in compression")
            return False

    # ── compression mode helper ───────────────────────────────────────────────

    def _resolve_compression_mode(self, n_messages: int):
        """
        Map user's compression_mode string to axor-core CompressionMode.
        In "auto" mode, derive from session length.
        """
        from axor_core.contracts.policy import CompressionMode
        if self._compression_mode == "auto":
            if n_messages <= 6:
                return CompressionMode.LIGHT
            elif n_messages <= 20:
                return CompressionMode.BALANCED
            else:
                return CompressionMode.AGGRESSIVE
        mapping = {
            "minimal":  CompressionMode.AGGRESSIVE,
            "moderate": CompressionMode.BALANCED,
            "broad":    CompressionMode.LIGHT,
        }
        return mapping.get(self._compression_mode, CompressionMode.BALANCED)

    def _resolve_context_mode(self, n_messages: int):
        from axor_core.contracts.policy import ContextMode
        if n_messages <= 6:
            return ContextMode.BROAD
        elif n_messages <= 20:
            return ContextMode.MODERATE
        else:
            return ContextMode.MINIMAL

    def _resolve_policy(self, messages: list):
        """Build a minimal ExecutionPolicy for this turn."""
        from axor_core.contracts.policy import (
            ExecutionPolicy, CompressionMode, ContextMode,
            ChildMode, ExportMode, ToolPolicy,
        )
        n = len(messages)
        return ExecutionPolicy(
            name="axor-langchain",
            context_mode=self._resolve_context_mode(n),
            compression_mode=self._resolve_compression_mode(n),
            child_mode=ChildMode.DENIED,
            export_mode=ExportMode.SUMMARY,
            tool_policy=ToolPolicy(
                allow_read=True,
                allow_write=True,
                allow_bash=True,
                allow_search=True,
            ),
        )

    # ── govern messages ───────────────────────────────────────────────────────

    def _govern_messages_core(self, messages: list) -> tuple[list, bool]:
        """
        Use axor-core ContextManager to compress messages.
        Returns (compressed_messages, changed).
        Sets self._cancelled if hard budget exceeded.
        """
        if not self._ensure_engines():
            return self._govern_messages_fallback(messages)

        from axor_core.contracts.context import RawExecutionState

        self._turn += 1

        # ── budget check via policy engine ────────────────────────────────────
        current_tokens = sum(_msg_tokens(m) for m in messages)
        total_spent = self._budget_tracker.total_tokens()

        if self._hard_limit and total_spent + current_tokens > self._hard_limit:
            if self._verbose:
                print(f"[axor] hard budget stop: {total_spent} tokens spent")
            self._cancelled = True
            return messages, False

        if self._soft_limit and total_spent > self._soft_limit and self._verbose:
            print(f"[axor] soft limit warning: {total_spent}/{self._soft_limit} tokens")

        # ── record this turn's input into budget tracker ───────────────────────
        self._budget_tracker.record(
            node_id=self._session_id,
            input_tokens=current_tokens,
            output_tokens=0,
        )

        # ── bypass: skip compression for small contexts ───────────────────────
        # Budget check and recording still happen above — only the
        # compression pipeline (fragments → compress → reconstruct) is skipped.
        if self._bypass_threshold and current_tokens < self._bypass_threshold:
            if self._verbose:
                print(f"[axor] bypass: {current_tokens} tokens < {self._bypass_threshold} threshold")
            return messages, False

        # ── build context via axor-core pipeline ──────────────────────────────
        task = _msg_text(messages[-1]) if messages else ""
        lineage = _make_lineage(self._session_id)
        policy  = self._resolve_policy(messages)

        # Convert messages → ContextFragments and add to manager
        fragments = _messages_to_fragments(messages)
        raw_state = RawExecutionState(
            task=task,
            session_id=self._session_id,
            parent_export=None,
            session_state={},
            memory_fragments=[],
            lineage=lineage,
            prior_turns=[],   # handled via fragments
        )
        # Feed all fragments as memory_fragments (plain strings)
        # and let ContextManager compress + select
        raw_state.memory_fragments = [f.content for f in fragments
                                       if f.value not in ("pinned",)]

        # Inject fragments via public API
        pinned = [f for f in fragments if f.value == "pinned"]
        non_pinned = [f for f in fragments if f.value != "pinned"]
        for frag in pinned:
            self._ctx_manager.pin_fragment(frag)
        if non_pinned:
            self._ctx_manager.ingest_fragments(non_pinned)

        context_view = self._ctx_manager.build(
            raw_state=raw_state,
            lineage=lineage,
            policy=policy,
        )

        if self._verbose:
            ratio = context_view.compression_ratio
            before = sum(_msg_tokens(m) for m in messages)
            after  = context_view.token_count
            if abs(before - after) > 10:
                print(
                    f"[axor] turn {self._turn}: {before}→{after} tokens "
                    f"(ratio={ratio:.2f}, mode={policy.compression_mode.value})"
                )

        compressed, changed = _fragments_to_messages(
            context_view, messages, self._personality
        )
        return compressed, changed

    def _govern_messages_fallback(self, messages: list) -> tuple[list, bool]:
        """
        Pure-Python fallback when axor-core is not installed.
        Simple window + tool truncation — same as original middleware.
        """
        self._turn += 1

        # bypass: skip compression for small contexts
        current_tokens = sum(_msg_tokens(m) for m in messages)
        if self._bypass_threshold and current_tokens < self._bypass_threshold:
            if self._verbose:
                print(f"[axor] bypass (fallback): {current_tokens} tokens < {self._bypass_threshold} threshold")
            return messages, False

        n = len(messages)
        if self._compression_mode == "auto":
            mode = "broad" if n <= 6 else "moderate" if n <= 20 else "minimal"
        else:
            mode = self._compression_mode

        system     = [m for m in messages if getattr(m, "type", None) == "system"]
        non_system = [m for m in messages if getattr(m, "type", None) != "system"]
        windows    = {"minimal": 6, "moderate": 16, "broad": 40}
        window     = windows.get(mode, 16)
        kept       = non_system[-window:] if len(non_system) > window else non_system
        max_tool   = {"minimal": 800, "moderate": 2000, "broad": 8000}.get(mode, 2000)

        compressed = []
        for msg in kept:
            content = getattr(msg, "content", "")
            if getattr(msg, "type", None) == "tool" and isinstance(content, str):
                if len(content) > max_tool:
                    head = content[:max_tool // 2]
                    tail = content[-(max_tool // 4):]
                    truncated = f"{head}\n…[truncated by axor]…\n{tail}"
                    try:
                        msg = msg.model_copy(update={"content": truncated})
                    except AttributeError:
                        import copy; msg = copy.copy(msg); msg.content = truncated
            compressed.append(msg)

        result = system + compressed

        # personality injection
        if self._personality:
            from langchain_core.messages import SystemMessage
            if system:
                merged = f"{self._personality}\n\n{_msg_text(system[0])}"
                try:
                    result[0] = system[0].model_copy(update={"content": merged})
                except AttributeError:
                    pass
            else:
                result.insert(0, SystemMessage(content=self._personality))

        total_spent = sum(_msg_tokens(m) for m in messages)
        if self._hard_limit and total_spent > self._hard_limit:
            self._cancelled = True

        return result, result != messages

    # ── tool filter ───────────────────────────────────────────────────────────

    def _filter_tools(self, tools: list) -> list:
        if not self._allowed_tools and not self._denied_tools:
            return tools
        filtered = []
        for tool in tools:
            name = getattr(tool, "name", None) or (
                tool.get("name") if isinstance(tool, dict) else None
            )
            if name:
                if self._allowed_tools and name not in self._allowed_tools:
                    if self._verbose:
                        print(f"[axor] denied (not allowed): {name}")
                    if self._tool_stats:
                        self._tool_stats.record_denied(name)
                    continue
                if name in self._denied_tools:
                    if self._verbose:
                        print(f"[axor] denied (blacklist): {name}")
                    if self._tool_stats:
                        self._tool_stats.record_denied(name)
                    continue
            filtered.append(tool)
        return filtered

    def _record_usage(self, response) -> None:
        usage = getattr(response, "usage_metadata", None)
        if usage and self._engines_ready:
            output_tokens = getattr(usage, "output_tokens", 0) or 0
            self._budget_tracker.record(
                node_id=self._session_id,
                input_tokens=0,
                output_tokens=output_tokens,
            )
            if self._verbose:
                total = self._budget_tracker.total_tokens()
                print(f"[axor] usage: +{output_tokens} out tokens (total: {total})")

    # ── tool execution ────────────────────────────────────────────────────────

    def _execute_tool_governed(self, tool_name: str, request, handler: Callable) -> Any:
        attempts   = self._tool_max_retries + 1
        last_error: Exception | None = None
        t0 = time.monotonic()

        for attempt in range(1, attempts + 1):
            try:
                result = handler(request)
                latency_ms = (time.monotonic() - t0) * 1000
                if self._tool_stats:
                    self._tool_stats.record_call(tool_name, latency_ms)
                if self._verbose:
                    print(f"[axor] → {tool_name!r} ok "
                          f"(attempt {attempt}, {latency_ms:.0f}ms)")
                return result
            except Exception as exc:
                last_error = exc
                if self._verbose:
                    print(f"[axor] → {tool_name!r} error "
                          f"(attempt {attempt}/{attempts}): {exc}")
                if attempt < attempts and self._tool_retry_delay > 0:
                    # sync sleep — LangChain's wrap_tool_call is a sync hook
                    time.sleep(self._tool_retry_delay)

        latency_ms = (time.monotonic() - t0) * 1000
        if self._tool_stats:
            self._tool_stats.record_call(tool_name, latency_ms)
            self._tool_stats.record_error(tool_name)

        if self._tool_error_handler is not None:
            msg = self._tool_error_handler(tool_name, last_error)
            if self._verbose:
                print(f"[axor] → {tool_name!r} handled error")
            return msg

        raise last_error

    # ── public API ────────────────────────────────────────────────────────────

    @property
    def total_tokens_spent(self) -> int:
        if self._engines_ready:
            return self._budget_tracker.total_tokens()
        return 0

    @property
    def turns(self) -> int:
        return self._turn

    @property
    def tool_stats(self) -> ToolStats | None:
        return self._tool_stats

    def reset(self) -> None:
        self._turn      = 0
        self._cancelled = False
        if self._tool_stats:
            self._tool_stats.reset()
        # reset engines so next session gets a fresh context manager
        self._ctx_manager    = None
        self._budget_tracker = None
        self._budget_engine  = None
        self._engines_ready  = False


# ── LangChain AgentMiddleware subclass ────────────────────────────────────────

def _make_axor_middleware_class():
    from langchain.agents.middleware import (
        AgentMiddleware,
        AgentState,
        ModelRequest,
        ModelResponse,
        ToolCallRequest,
        hook_config,
    )
    from langchain_core.messages import ToolMessage, SystemMessage
    from langgraph.runtime import Runtime

    class AxorMiddlewareImpl(_AxorGovernanceCore, AgentMiddleware):

        def __init__(self, **kwargs) -> None:
            _AxorGovernanceCore.__init__(self, **kwargs)
            AgentMiddleware.__init__(self)

        @hook_config(can_jump_to=["end"])
        def before_model(
            self,
            state: AgentState,
            runtime: Runtime,
        ) -> dict[str, Any] | None:
            if self._cancelled:
                return {"jump_to": "end"}

            messages = list(state.get("messages", []))
            new_messages, changed = self._govern_messages_core(messages)

            if self._cancelled:
                return {"jump_to": "end"}

            if changed:
                return {"messages": new_messages}
            return None

        def wrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], ModelResponse],
        ) -> ModelResponse:
            # filter tools by policy
            tools = getattr(request, "tools", None) or []
            filtered = self._filter_tools(tools)
            if len(filtered) != len(tools):
                request = request.override(tools=filtered)

            response = handler(request)
            self._record_usage(response)
            return response

        def wrap_tool_call(
            self,
            request: ToolCallRequest,
            handler: Callable[[ToolCallRequest], Any],
        ) -> Any:
            tc        = request.tool_call if hasattr(request, "tool_call") else {}
            tool_name = _tool_name(tc)
            tc_id     = _tool_call_id(tc)

            if self._verbose:
                args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
                print(f"[axor] → tool {tool_name!r} args={args}")

            result = self._execute_tool_governed(tool_name, request, handler)

            # wrap string error returns in ToolMessage
            if isinstance(result, str) and self._tool_error_handler is not None:
                return ToolMessage(content=result, tool_call_id=tc_id or "")

            # record tool output size to budget tracker for accurate accounting
            if self._engines_ready:
                out_text = result if isinstance(result, str) else str(
                    getattr(result, "content", "") or ""
                )
                self._budget_tracker.record(
                    node_id=self._session_id,
                    input_tokens=0,
                    output_tokens=0,
                    tool_tokens=max(len(out_text) // 4, 1),
                )
                # also feed result into ContextManager for next turn compression
                self._ctx_manager.update(
                    result_output=out_text[:2000],
                    node_id=self._session_id,
                )

            return result

        def after_agent(
            self,
            state: AgentState,
            runtime: Runtime,
        ) -> dict[str, Any] | None:
            if self._verbose and self._tool_stats:
                print(f"[axor] {self._tool_stats.summary()}")
            if self._verbose and self._engines_ready:
                print(f"[axor] session total tokens: {self._budget_tracker.total_tokens()}")
            # sync hook — memory save handled in aafter_agent
            return None

        async def aafter_agent(
            self,
            state: AgentState,
            runtime: Runtime,
        ) -> dict[str, Any] | None:
            self.after_agent(state, runtime)
            await self._record_telemetry(state)
            if self._memory_provider is None:
                return None
            try:
                from axor_core.contracts.memory import MemoryFragment, FragmentValue
                from datetime import datetime, timezone
                messages = state.get("messages", [])
                ai_msgs  = [m for m in messages if getattr(m, "type", None) == "ai"]
                if ai_msgs:
                    content = _msg_text(ai_msgs[-1])
                    if content:
                        await self._memory_provider.save([MemoryFragment(
                            namespace=self._memory_namespace,
                            key=f"turn_{self._turn}_{datetime.now(timezone.utc).timestamp():.0f}",
                            content=content[:2000],
                            value=FragmentValue.WORKING,
                        )])
            except Exception as e:
                if self._verbose:
                    print(f"[axor] memory save failed: {e}")
            return None

        async def _record_telemetry(self, state) -> None:
            """
            Classify the latest human message with axor-core's TaskAnalyzer
            and forward one AnonymizedTraceRecord to the telemetry sink.
            Silent no-op if telemetry is off or axor-core is not available.
            """
            if self._telemetry is None or not getattr(self._telemetry, "enabled", False):
                return
            try:
                messages = state.get("messages", [])
                human = None
                for m in messages:
                    if getattr(m, "type", None) == "human":
                        human = _msg_text(m)
                if not human:
                    return
                from axor_core.policy.analyzer import TaskAnalyzer
                analyzer = TaskAnalyzer()
                signal, event = await analyzer.analyze(human)
                tokens = self._budget_tracker.total_tokens() if self._engines_ready else 0
                await self._telemetry.record_decision(
                    raw_input=human,
                    signal=signal,
                    classifier_used=event.classifier,
                    confidence=float(event.confidence),
                    tokens_spent=int(tokens),
                    policy_adjusted=False,
                )
            except Exception as e:
                if self._verbose:
                    print(f"[axor] telemetry record failed: {e}")

    AxorMiddlewareImpl.__name__ = "AxorMiddleware"
    return AxorMiddlewareImpl


# ── Public AxorMiddleware ──────────────────────────────────────────────────────

class AxorMiddleware(_AxorGovernanceCore):
    """
    Governance middleware for LangChain 1.0 agents backed by axor-core engines.

    When axor-core is installed, uses:
      • ContextManager + ContextCompressor  — fragment-aware compression with
        pinned/knowledge/working/ephemeral fragment semantics
      • BudgetTracker + BudgetPolicyEngine  — accurate per-turn token accounting
        with soft warnings and hard stops

    Falls back to built-in simple windowed compression when axor-core is absent.

    Parameters
    ----------
    soft_token_limit : int | None
        Advisory threshold — logs a warning when crossed.
    hard_token_limit : int | None
        Hard stop — agent jumps to "end" when crossed.
        Defaults to soft_token_limit * 1.5 when only soft is set.
    compression_mode : str
        "auto" | "minimal" | "moderate" | "broad"
        Maps to axor-core AGGRESSIVE / BALANCED / LIGHT respectively.
    allowed_tools : list[str] | None
        Whitelist — only these tools reach the model.
    denied_tools : list[str] | None
        Blacklist — these tools are stripped from every model call.
    personality : str | None
        Pinned system message — survives all compression.
    memory_provider : MemoryProvider | None
        axor-core memory provider for cross-session persistence.
    memory_namespace : str
        Namespace for memory writes (default: "axor").
    tool_error_handler : Callable[[str, Exception], str] | None
        Called on final tool failure. Return value becomes a ToolMessage.
    tool_max_retries : int
        Extra retry attempts after first failure (default: 0).
    tool_retry_delay : float
        Seconds between retries (default: 0.0).
    track_tool_stats : bool
        Enable per-tool call/latency/error tracking (access via .tool_stats).
    verbose : bool
        Print governance decisions to stdout.

    Examples
    --------
    Basic::

        axor = AxorMiddleware(soft_token_limit=80_000, verbose=True)
        agent = create_agent("anthropic:claude-sonnet-4-5", tools=tools, middleware=[axor])

    With retry and error handling::

        def on_err(name, exc): return f"Tool {name!r} failed: {exc}"
        axor = AxorMiddleware(tool_max_retries=2, tool_error_handler=on_err)

    Per-agent governance in LangGraph::

        research = AxorMiddleware(allowed_tools=["search"], soft_token_limit=50_000)
        writer   = AxorMiddleware(allowed_tools=["read", "write"], soft_token_limit=30_000)
    """

    _impl_class = None

    def __new__(cls, **kwargs):
        if cls._impl_class is None:
            try:
                cls._impl_class = _make_axor_middleware_class()
            except ImportError:
                cls._impl_class = None

        if cls._impl_class is not None:
            instance = object.__new__(cls._impl_class)
            # Python skips __init__ when __new__ returns a different type,
            # so we must initialize explicitly
            cls._impl_class.__init__(instance, **kwargs)
            return instance
        return object.__new__(cls)

    def __init__(
        self,
        soft_token_limit: int | None = None,
        hard_token_limit: int | None = None,
        compression_mode: str = "auto",
        bypass_token_threshold: int | None = None,
        allowed_tools: list[str] | None = None,
        denied_tools:  list[str] | None = None,
        personality:   str | None = None,
        memory_provider=None,
        memory_namespace: str = "axor",
        tool_error_handler: Callable[[str, Exception], str] | None = None,
        tool_max_retries: int = 0,
        tool_retry_delay: float = 0.0,
        track_tool_stats: bool = False,
        verbose: bool = False,
        telemetry: str | None = None,
    ) -> None:
        # Guard against double init — __new__ already called __init__
        # on AxorMiddlewareImpl instances
        if getattr(self, "_engines_ready", None) is not None:
            return
        _AxorGovernanceCore.__init__(
            self,
            soft_token_limit=soft_token_limit,
            hard_token_limit=hard_token_limit,
            compression_mode=compression_mode,
            bypass_token_threshold=bypass_token_threshold,
            allowed_tools=allowed_tools,
            denied_tools=denied_tools,
            personality=personality,
            memory_provider=memory_provider,
            memory_namespace=memory_namespace,
            tool_error_handler=tool_error_handler,
            tool_max_retries=tool_max_retries,
            tool_retry_delay=tool_retry_delay,
            track_tool_stats=track_tool_stats,
            verbose=verbose,
            telemetry=telemetry,
        )
