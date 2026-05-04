from __future__ import annotations

"""
axor-langchain — governance middleware for LangChain 1.0 agents.

Single hook (`wrap_model_call`) that does, in order:

  1. Filter tools by allow/deny lists.
  2. Merge optional `personality` into `request.system_message` (plain
     prepend; provider caching is delegated to
     `langchain_anthropic.AnthropicPromptCachingMiddleware`).
  3. Compress `request.messages` via `axor_core.context.ContextCompressor`,
     using FragmentValue assignment from `_classify_messages` and
     CompressionMode picked by `axor_core.policy.PolicySelector` from the
     latest user task.
  4. Pre-call hard-budget gate: raises `axor_core.errors.BudgetExceededError`
     when the projected input would exceed `hard_token_limit`.
  5. Hand off to the underlying handler.
  6. Record provider-counted input + output tokens to `BudgetTracker`.

`wrap_tool_call` adds:
  • Optional result memoization for explicitly opted-in tools, keyed by
    `hash(tool_name, args)` via the `tool_result_cache` field on a custom
    `AxorState`. LangGraph checkpointing persists hits across
    `agent.invoke()` calls under the same `thread_id`.
  • Retry + error handler + per-tool stats.

Everything is opt-in beyond the default constructor; with no kwargs
AxorMiddleware is a thin pass-through that records token usage.
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

_log = logging.getLogger("axor.langchain")


_TELEMETRY_MARKER = Path.home() / ".axor" / ".telemetry_notice_shown"


def _maybe_show_telemetry_notice() -> None:
    """Print a one-line stderr notice on first construction. Idempotent."""
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


_missing_telemetry_warned = False


def _warn_missing_telemetry(mode: str) -> None:
    global _missing_telemetry_warned
    if _missing_telemetry_warned or os.environ.get("AXOR_NO_BANNER") == "1":
        return
    sys.stderr.write(
        f"axor: telemetry={mode!r} requested but axor-telemetry is not installed. "
        "Install with: pip install axor-langchain[telemetry] "
        "(or set AXOR_NO_BANNER=1 to silence)\n"
    )
    _missing_telemetry_warned = True


def _build_telemetry_pipeline(mode: str, axor_version: str) -> Any | None:
    if mode == "off":
        return None
    try:
        from axor_telemetry import TelemetryConfig, TelemetryMode, build_pipeline
    except ImportError:
        _warn_missing_telemetry(mode)
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
    """Extract plain text from a LangChain message (handles list-of-blocks)."""
    content = getattr(msg, "content", "") or ""
    if isinstance(content, list):
        return " ".join(
            b.get("text", "") for b in content if isinstance(b, dict)
        )
    return str(content)


# Conservative per-block estimates for non-text content. Anthropic vision
# bills around ~1.5k tokens for a typical image; document blocks are
# similar. These are deliberately on the high side so the hard-budget gate
# doesn't *underestimate* a multimodal payload and let it through.
_NONTEXT_BLOCK_ESTIMATE = {
    "image":           1500,
    "input_image":     1500,
    "image_url":       1500,
    "document":        1500,
    "file":            1500,
    "tool_use":        50,
    "tool_result":     200,
    "input_audio":     2000,
}


def _msg_tokens(msg) -> int:
    """Estimate token cost of a single message, including non-text blocks.

    Previously this only counted text characters // 4, which reported a
    1-token cost for an attached image — letting multimodal payloads slip
    past the hard-budget gate before reaching the provider.
    """
    content = getattr(msg, "content", "") or ""
    text_chars = 0
    nontext_tokens = 0

    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                btype = block.get("type") or ""
                if btype in ("text", ""):
                    text_chars += len(block.get("text", "") or "")
                else:
                    nontext_tokens += _NONTEXT_BLOCK_ESTIMATE.get(btype, 200)
            else:
                # Object-style block; best-effort fallback.
                btype = getattr(block, "type", "") or ""
                txt = getattr(block, "text", "")
                if btype in ("text", ""):
                    text_chars += len(txt or "")
                else:
                    nontext_tokens += _NONTEXT_BLOCK_ESTIMATE.get(btype, 200)
    else:
        text_chars = len(str(content))

    return max(text_chars // 4 + nontext_tokens, 1)


def _tool_name(tool_call) -> str:
    if isinstance(tool_call, dict):
        return tool_call.get("name", "unknown")
    return getattr(tool_call, "name", "unknown")


def _tool_call_id(tool_call) -> str:
    if isinstance(tool_call, dict):
        return tool_call.get("id", "")
    return getattr(tool_call, "id", "")


def _hash_tool_call(tool_name: str, args: Any) -> str:
    """
    Stable hash of (tool_name, args) for tool-result-cache lookup.

    Properties:
      • Order-invariant for dict args: hash({a:1,b:2}) == hash({b:2,a:1})
      • Same value on every call (for cross-invocation cache via
        LangGraph checkpoint replay)
      • Never raises — non-JSON-serializable args fall back to repr()

    Used as the key in `state["tool_result_cache"]`. Collisions are
    cryptographically unlikely (md5 hex, 128 bits).
    """
    try:
        if isinstance(args, dict):
            payload = json.dumps(
                [tool_name, sorted(args.items())],
                default=repr, sort_keys=True,
            )
        else:
            payload = json.dumps([tool_name, args], default=repr)
    except (TypeError, ValueError):
        payload = repr((tool_name, args))
    return hashlib.md5(payload.encode("utf-8", errors="replace")).hexdigest()


def _is_tool_call_cacheable(tool_name: str, cache_tools: set[str] | None) -> bool:
    """Tool result caching is opt-in because LangChain tools may have side effects."""
    return cache_tools is not None and tool_name in cache_tools


class _HandledError(str):
    """
    Marker subclass returned by the governance helpers when `tool_error_handler`
    converted an exception into a string. Successful string-returning tools
    yield a plain `str`; this distinction is what the cache write path uses to
    skip caching errors without skipping legitimate results.
    """


class AxorCancelledError(RuntimeError):
    """Raised by the middleware when `cancel()` was called externally.

    Surfaces as a regular exception out of the agent's `invoke`/`ainvoke`,
    letting the caller distinguish a user-initiated abort from a model
    error. Catch it at the agent boundary if you want graceful shutdown.
    """


# ── Per-tool stats ─────────────────────────────────────────────────────────────

@dataclass
class ToolCallStats:
    call_count: int = 0
    error_count: int = 0
    total_latency_ms: float = 0.0
    denied_count: int = 0
    cache_hits: int = 0   # served from tool_result_cache, no execution

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

    @property
    def cache_hit_rate(self) -> float:
        """Fraction of invocations served from cache without re-executing."""
        total = self.call_count + self.cache_hits
        return self.cache_hits / total if total else 0.0


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

    def record_cache_hit(self, name: str) -> None:
        self._get(name).cache_hits += 1

    @property
    def total_calls(self) -> int:
        return sum(s.call_count for s in self.by_tool.values())

    @property
    def total_errors(self) -> int:
        return sum(s.error_count for s in self.by_tool.values())

    @property
    def total_cache_hits(self) -> int:
        return sum(s.cache_hits for s in self.by_tool.values())

    def summary(self) -> str:
        lines = [
            f"Tool calls: {self.total_calls} executed, "
            f"{self.total_cache_hits} cache hits, "
            f"{self.total_errors} errors, "
            f"{sum(s.denied_count for s in self.by_tool.values())} denied"
        ]
        for name, s in sorted(self.by_tool.items()):
            cache_str = f", {s.cache_hits} cached" if s.cache_hits else ""
            lines.append(
                f"  {name}: {s.call_count} calls{cache_str}, "
                f"{s.avg_latency_ms:.0f}ms avg, "
                f"{s.success_rate:.0%} success"
            )
        return "\n".join(lines)

    def reset(self) -> None:
        self.by_tool.clear()


# FragmentValue labels control what axor-core may compress or preserve.
# Decision detection delegates to axor-core to avoid regex drift.
def _has_decision_content(text: str) -> bool:
    """Delegate to axor-core's `has_decision_content` (single source of truth)."""
    try:
        from axor_core.context.compressor import has_decision_content
        return has_decision_content(text)
    except ImportError:
        # Minimal fallback when axor-core's helper is unavailable.
        return bool(re.search(
            r"\b(decided|chose|using|will use|replaced)\b",
            text or "", re.IGNORECASE,
        ))


def _classify_messages(
    messages: list,
    *,
    recent_tools_window: int = 2,
):
    """
    Assign a FragmentValue from axor-core to every message.

    No compression here — just labelling. Returns a list of (msg, value)
    pairs in the same order as the input. Output feeds
    `_messages_to_fragments` and then `axor_core.context.ContextCompressor`
    which respects FragmentValue semantics (PINNED untouched, etc).

    Parameters
    ----------
    messages : list
        LangChain messages. Order matters — last item is the most recent.
    recent_tools_window : int, default 2
        Tool messages within this many positions of the end stay WORKING
        (visible to the model with normal compression). Older ones go
        EPHEMERAL (aggressive truncation candidate). Default 2 covers
        typical parallel-tool-call patterns.
    """
    from axor_core.contracts.memory import FragmentValue

    n = len(messages)

    # Latest AI tool-call message anchors the currently pending tool result.
    latest_ai_tool_call_idx = None
    for i in range(n - 1, -1, -1):
        m = messages[i]
        if getattr(m, "type", None) == "ai":
            tool_calls = getattr(m, "tool_calls", None) or []
            if tool_calls:
                latest_ai_tool_call_idx = i
                break

    tool_indices = [
        i for i, m in enumerate(messages)
        if getattr(m, "type", None) == "tool"
    ]
    recent_tool_set = (
        set(tool_indices[-recent_tools_window:])
        if recent_tools_window > 0 else set()
    )

    result: list = []
    for i, msg in enumerate(messages):
        mtype = getattr(msg, "type", None)

        if mtype == "system":
            value = FragmentValue.PINNED

        elif mtype == "human":
            value = FragmentValue.PINNED

        elif mtype == "ai":
            tool_calls = getattr(msg, "tool_calls", None) or []
            if tool_calls and i == latest_ai_tool_call_idx:
                value = FragmentValue.PINNED
            elif tool_calls:
                value = FragmentValue.WORKING
            else:
                content = _msg_text(msg)
                if _has_decision_content(content):
                    value = FragmentValue.KNOWLEDGE
                else:
                    value = FragmentValue.WORKING

        elif mtype == "tool":
            # Fresh tool outputs stay verbatim; old ones are compression targets.
            value = (
                FragmentValue.PINNED if i in recent_tool_set
                else FragmentValue.EPHEMERAL
            )

        else:
            value = FragmentValue.WORKING

        result.append((msg, value))

    return result


# HeuristicClassifier is async by protocol, but its internals are sync.
# Calling internals here avoids event-loop conflicts inside LangGraph.

def _classify_task_sync(text: str):
    """
    Classify a user task synchronously. Returns axor-core `TaskSignal`.
    Empty / unclassifiable text returns the FOCUSED/GENERATIVE default
    (matches HeuristicClassifier's own fallback behavior).
    """
    from axor_core.policy.heuristic import HeuristicClassifier
    from axor_core.contracts.policy import (
        TaskSignal, TaskComplexity, TaskNature,
    )
    cls = HeuristicClassifier()
    text_stripped = (text or "").strip()
    complexity, _ = cls._classify_complexity(text_stripped)
    nature, _ = cls._classify_nature(text_stripped)
    return TaskSignal(
        raw_input=text or "",
        complexity=complexity,
        nature=nature,
        estimated_scope=cls._estimate_scope(complexity),
        requires_children=complexity == TaskComplexity.EXPANSIVE,
        requires_mutation=nature == TaskNature.MUTATIVE,
    )


def _normalize_compression_mode_name(mode: str | None) -> str | None:
    if mode is None:
        return None
    normalized = str(mode).strip().lower().replace("-", "_")
    aliases = {
        "auto": None,
        "aggressive": "AGGRESSIVE",
        "minimal": "AGGRESSIVE",
        "balanced": "BALANCED",
        "moderate": "BALANCED",
        "light": "LIGHT",
        "broad": "LIGHT",
    }
    if normalized not in aliases:
        allowed = ", ".join(sorted(k for k in aliases if k != "auto"))
        raise ValueError(
            f"unsupported compression_mode={mode!r}; use auto, {allowed}"
        )
    return aliases[normalized]


# ── Tool relevance gating ──────────────────────────────────────────────────────
#
# Replaces hand-curated `allowed_tools=[...]` with dynamic per-call selection
# based on axor-core-style keyword overlap with the latest user task. Rationale
# is in CLAUDE.md and the README benchmark notes: "savings should come from
# middleware logic, not from human-curated per-task tool subsets."
#
# Quality preservation:
#   • Tools mentioned in pending tool_calls (last AI hasn't received tool_result
#     yet) are NEVER dropped — provider APIs reject orphan tool_call/tool_result.
#   • Tools called within the last `tool_sticky_lookback` AI turns are anchored
#     in the keep-set even if their relevance score is low this round.
#   • A floor of `tool_min_keep` prevents over-zealous trimming on edge cases.
#
# The vocabulary, scoring, synonym expansion, and topic-strength logic are
# all provider-agnostic — they live in `axor_core.policy.keyword_relevance`
# so axor-claude / axor-openai adapters can reuse the same heuristic
# without copy-paste. This module keeps only LangChain-specific bits:
#   • `_pending_tool_call_names`, `_recently_called_tool_names` —
#     anchor extraction from a LangChain message list.
#   • `_select_relevant_tools` — orchestration that combines core scoring
#     with anchor logic and the floor/top-k policy knobs.
#   • `_tool_attr` — duck-types LangChain tools (BaseTool, dict, …).

from axor_core.policy import topics as _topics
from axor_core.policy.keyword_relevance import (
    compute_topic_strength as _compute_topic_strength,
    expand_with_synonyms as _expand_with_synonyms,
    extract_query_keywords as _extract_query_keywords,
    name_has_destructive_token as _name_has_destructive_token,
    score_tool_relevance as _score_core,
    tool_topics as _tool_topics_for_name,
)


# Re-export under legacy private names so existing tests and any
# downstream code that imported these directly keep working.
_DOMAIN_TOPICS = _topics.DOMAIN_TOPICS
_TOPIC_IMPLICATIONS = _topics.TOPIC_IMPLICATIONS
_WORD_TOPICS = _topics.WORD_TOPICS
_SYNONYM_MAP = _topics.SYNONYM_MAP
_STOPWORDS = _topics.STOPWORDS
_DESTRUCTIVE_TOKENS = _topics.DESTRUCTIVE_TOKENS


def _tool_attr(tool, attr: str, default: str = "") -> str:
    """Read tool.{attr} or tool[attr] uniformly across LangChain BaseTool
    instances and plain dict-shaped tool descriptors."""
    if isinstance(tool, dict):
        v = tool.get(attr, default)
    else:
        v = getattr(tool, attr, default)
    return str(v or default)


def _tool_topics(tool) -> set[str]:
    """LangChain-side adapter — extracts the tool name and delegates to
    `axor_core.policy.tool_topics(name)` for actual topic membership."""
    return _tool_topics_for_name(_tool_attr(tool, "name"))


def _score_tool_relevance(
    tool, keywords: set[str], *,
    use_synonyms: bool = True,
    topic_strength: dict[str, float] | None = None,
) -> float:
    """LangChain-side adapter for `axor_core.policy.score_tool_relevance`.
    Extracts name + description from the duck-typed tool and forwards."""
    return _score_core(
        name=_tool_attr(tool, "name"),
        description=_tool_attr(tool, "description"),
        keywords=keywords,
        use_synonyms=use_synonyms,
        topic_strength=topic_strength,
    )


def _pending_tool_call_names(messages: list) -> set[str]:
    """
    Names of tools whose tool_call has been emitted but no matching
    tool_result has come back. Their schema MUST stay in the request,
    otherwise the provider rejects the orphan tool_call.
    """
    pending: dict[str, str] = {}  # tool_call_id → tool_name
    for m in messages:
        mtype = getattr(m, "type", None)
        if mtype == "ai":
            for tc in getattr(m, "tool_calls", None) or []:
                tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", "")
                tc_name = _tool_name(tc)
                if tc_id:
                    pending[str(tc_id)] = tc_name
        elif mtype == "tool":
            tc_id = str(getattr(m, "tool_call_id", "") or "")
            pending.pop(tc_id, None)
    return set(pending.values())


def _recently_called_tool_names(messages: list, lookback: int) -> set[str]:
    """Names of tools the agent called in the last `lookback` AI rounds."""
    if lookback <= 0:
        return set()
    seen_ai = 0
    sticky: set[str] = set()
    for m in reversed(messages):
        if getattr(m, "type", None) != "ai":
            continue
        for tc in getattr(m, "tool_calls", None) or []:
            sticky.add(_tool_name(tc))
        seen_ai += 1
        if seen_ai >= lookback:
            break
    return sticky


def _normalize_tool_selection(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in ("", "none", "off"):
        return None
    if normalized not in ("relevance",):
        raise ValueError(
            f"unsupported tool_selection={value!r}; use None or 'relevance'"
        )
    return normalized


_OPTIMIZATION_PROFILES: dict[str, dict[str, Any]] = {
    "cautious": {
        "recent_tools_window": 2,
        "compression_mode": None,
        "tool_selection": None,
    },
    "aggressive": {
        "recent_tools_window": 1,
        "compression_mode": "aggressive",
        "tool_selection": "relevance",
        "tool_top_k": 8,
        "tool_dedup_old_results": True,
    },
}


def _normalize_optimization_profile(profile: str | None) -> str | None:
    if profile is None:
        return None
    normalized = str(profile).strip().lower().replace("-", "_")
    if normalized in ("", "none", "custom"):
        return None
    if normalized not in _OPTIMIZATION_PROFILES:
        allowed = ", ".join(sorted(_OPTIMIZATION_PROFILES))
        raise ValueError(
            f"unsupported optimization_profile={profile!r}; use {allowed}"
        )
    return normalized


def _latest_human_text(messages: list) -> str:
    """Pull the last HumanMessage's text, or '' if none."""
    for m in reversed(messages):
        if getattr(m, "type", None) == "human":
            return _msg_text(m)
    return ""


# Compressor output is resorted by `source` so tool_call/tool_result order holds.
# Empty AI tool-call messages get a synthetic marker to survive deduplication.
_AI_TOOL_CALL_PREFIX = "[axor:tcs:"


def _build_fragment_content(msg) -> str:
    """
    Pre-compress text view of a LangChain message, with a unique synthetic
    marker for AIMessages that have empty content but carry tool_calls
    (otherwise the dedup strategy would collapse them all into one).
    """
    text = _msg_text(msg)
    mtype = getattr(msg, "type", None)
    if mtype == "ai" and not text.strip():
        tool_calls = getattr(msg, "tool_calls", None) or []
        if tool_calls:
            ids = [
                tc.get("id") if isinstance(tc, dict)
                else getattr(tc, "id", "")
                for tc in tool_calls
            ]
            return _AI_TOOL_CALL_PREFIX + ",".join(str(i) for i in ids if i) + "]"
    return text


def _build_dedup_overrides(
    messages: list, recent_tools_window: int,
) -> dict[int, str]:
    """
    For ToolMessages whose `(tool_name, args, content)` repeats an earlier
    call AND that fall outside the recent-tools window, return short reference
    content to use instead of their original payload. Prevents the same 10kB
    metric query from sitting in the prompt verbatim three times.

    Content equality is required: time-varying tools (metric/log queries,
    ticket lookups, file reads after edits) can return different evidence for
    identical args. Matching on args alone would silently replace fresh
    results with a pointer to a stale one.
    """
    if recent_tools_window < 0:
        recent_tools_window = 0

    tc_id_to_hash: dict[str, str] = {}
    for m in messages:
        if getattr(m, "type", None) != "ai":
            continue
        for tc in getattr(m, "tool_calls", None) or []:
            tc_id = (tc.get("id") if isinstance(tc, dict)
                     else getattr(tc, "id", "")) or ""
            if not tc_id:
                continue
            name = _tool_name(tc)
            args = (tc.get("args") if isinstance(tc, dict)
                    else getattr(tc, "args", {})) or {}
            tc_id_to_hash[str(tc_id)] = _hash_tool_call(name, args)

    tool_indices = [
        i for i, m in enumerate(messages)
        if getattr(m, "type", None) == "tool"
    ]
    recent_set = (
        set(tool_indices[-recent_tools_window:])
        if recent_tools_window > 0 else set()
    )

    # (args_hash, content) → tool_call_id of first occurrence. Content is
    # part of the key so that two calls with the same args but different
    # results are NOT collapsed into a pointer.
    seen_first_id: dict[tuple[str, str], str] = {}
    overrides: dict[int, str] = {}
    for i in tool_indices:
        m = messages[i]
        tc_id = str(getattr(m, "tool_call_id", "") or "")
        h = tc_id_to_hash.get(tc_id)
        if h is None:
            continue
        key = (h, _msg_text(m))
        first = seen_first_id.get(key)
        if first is None:
            seen_first_id[key] = tc_id
            continue
        if i in recent_set:
            continue
        overrides[i] = (
            f"[axor: duplicate tool call; identical args and result to "
            f"tool_call_id={first} — see prior result]"
        )
    return overrides


def _messages_to_fragments(
    messages: list,
    *,
    recent_tools_window: int = 2,
    dedup_old_results: bool = False,
):
    """
    Build ContextFragments from LangChain messages. Source field encodes
    the original position so we can reassemble in order after compression.

    `kind`:
      • system / human → "fact"
      • ai with tool_calls → "ai_tool_call" (custom kind; compressor leaves
        it alone other than the value-based grouping)
      • ai prose → "assistant_prose" (matches what _compress_prose looks for)
      • tool → "tool_result" (matches what _truncate_tool_outputs looks for)

    `turn` field is set to position+1 so axor-core's age computations
    (current_turn - turn) work meaningfully without us tracking a separate
    monotonic turn counter.

    `dedup_old_results=True` replaces the *content* of older ToolMessages
    whose (tool_name, args) duplicates an earlier call with a short pointer.
    Recent-window tool messages are never deduplicated.
    """
    from axor_core.contracts.context import ContextFragment

    classified = _classify_messages(
        messages,
        recent_tools_window=recent_tools_window,
    )
    overrides = (
        _build_dedup_overrides(messages, recent_tools_window)
        if dedup_old_results else {}
    )
    fragments = []
    for idx, (msg, value) in enumerate(classified):
        mtype = getattr(msg, "type", None)
        content = (
            overrides[idx] if idx in overrides
            else _build_fragment_content(msg)
        )
        if mtype == "ai":
            tool_calls = getattr(msg, "tool_calls", None) or []
            # Tool-call metadata must remain paired with its ToolMessage.
            kind = "ai_tool_call" if tool_calls else "assistant_prose"
        elif mtype == "tool":
            kind = "tool_result"
        else:  # system, human, unknown
            kind = "fact"

        fragments.append(ContextFragment(
            kind=kind,
            content=content,
            token_estimate=max(len(content) // 4, 1),
            source=f"lc:{idx}",
            relevance=1.0,
            value=value.value,
            turn=idx + 1,
        ))
    return fragments


def _parse_lc_source_index(source: str) -> int | None:
    """Extract the original message index from a fragment source string."""
    if not source.startswith("lc:"):
        return None
    try:
        return int(source[3:])
    except ValueError:
        return None


def _message_tool_call_ids(msg) -> set[str]:
    ids: set[str] = set()
    for tc in getattr(msg, "tool_calls", None) or []:
        tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", "")
        if tc_id:
            ids.add(str(tc_id))
    return ids


def _fragments_to_messages(compressed_fragments, original_messages: list):
    """
    Map compressed fragments back to LangChain messages.

    Strategy:
      • Index fragments by their `lc:{idx}` source.
      • Walk original_messages in order; emit each message that survived
        (with content possibly updated by truncate / prose-cap strategies).
      • Drop original messages whose fragment was eliminated by dedup or
        empty-removal.
      • Synthesized fragments (e.g. `compressor:prose_summary` from
        `_compress_prose`) get inserted as a HumanMessage right after the
        system + first-human prefix, so the model sees them as authoritative
        context. This mirrors langchain's SummarizationMiddleware idiom.
      • AI tool_call messages are returned verbatim — the synthetic content
        marker we put in their fragments is just a dedup anchor, not real
        content to be propagated back.
    """
    by_idx: dict[int, Any] = {}
    synthesized: list = []
    for f in compressed_fragments:
        idx = _parse_lc_source_index(f.source)
        if idx is not None:
            by_idx[idx] = f
        else:
            synthesized.append(f)

    out_by_idx: dict[int, Any] = {}
    for i, orig in enumerate(original_messages):
        if i not in by_idx:
            # Never drop AI tool-call messages; providers require paired results.
            if (getattr(orig, "type", None) == "ai"
                    and (getattr(orig, "tool_calls", None) or [])):
                out_by_idx[i] = orig
            continue
        f = by_idx[i]
        if f.kind == "ai_tool_call":
            out_by_idx[i] = orig
            continue
        original_text = _msg_text(orig)
        if f.content == original_text:
            out_by_idx[i] = orig
        else:
            # Preserve message metadata; only compressed content changes.
            try:
                new_msg = orig.model_copy(update={"content": f.content})
            except AttributeError:
                import copy
                new_msg = copy.copy(orig)
                try:
                    new_msg.content = f.content
                except (AttributeError, TypeError):
                    new_msg = orig
            out_by_idx[i] = new_msg

    # Restore required ToolMessages if compression removed a paired result.
    required_tool_ids: set[str] = set()
    for msg in out_by_idx.values():
        if getattr(msg, "type", None) == "ai":
            required_tool_ids.update(_message_tool_call_ids(msg))
    existing_tool_ids = {
        str(getattr(msg, "tool_call_id", ""))
        for msg in out_by_idx.values()
        if getattr(msg, "type", None) == "tool" and getattr(msg, "tool_call_id", "")
    }
    missing_tool_ids = required_tool_ids - existing_tool_ids
    if missing_tool_ids:
        for i, orig in enumerate(original_messages):
            if getattr(orig, "type", None) != "tool":
                continue
            tc_id = str(getattr(orig, "tool_call_id", ""))
            if tc_id in missing_tool_ids:
                out_by_idx[i] = orig

    out = [out_by_idx[i] for i in sorted(out_by_idx)]

    if synthesized:
        try:
            from langchain_core.messages import HumanMessage
        except ImportError:
            return out
        synth_msgs = [
            HumanMessage(
                content=f"[axor: condensed prior context]\n\n{f.content}",
                additional_kwargs={"lc_source": "axor.compress_prose"},
            )
            for f in synthesized
        ]
        insert_idx = 0
        seen_first_human = False
        for i, m in enumerate(out):
            mtype = getattr(m, "type", None)
            if mtype == "system":
                insert_idx = i + 1
            elif mtype == "human" and not seen_first_human:
                insert_idx = i + 1
                seen_first_human = True
            else:
                break
        out = out[:insert_idx] + synth_msgs + out[insert_idx:]

    return out


from typing import NotRequired
from langchain.agents.middleware import AgentState


class AxorState(AgentState):
    # hash(tool_name + args) -> cached content persisted by LangGraph state.
    tool_result_cache: NotRequired[dict[str, str]]


class _AxorGovernanceCore:
    """
    Pure governance logic. No LangChain imports here so this module is
    importable without langchain (helpful for unit tests of helpers).
    """

    def __init__(
        self,
        soft_token_limit: int | None = None,
        hard_token_limit: int | None = None,
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
        cache_tools: list[str] | None = None,
        max_tool_cache_entries: int | None = 100,
        recent_tools_window: int | None = None,
        compression_mode: str | None = None,
        optimization_profile: str | None = None,
        tool_selection: str | None = None,
        tool_top_k: int | None = None,
        tool_min_keep: int = 3,
        tool_sticky_lookback: int = 4,
        tool_dedup_old_results: bool | None = None,
        tool_selection_stable: bool = True,
        token_cost_rates: Any | None = None,
    ) -> None:
        profile = _normalize_optimization_profile(optimization_profile)
        profile_cfg = _OPTIMIZATION_PROFILES.get(profile or "", {})
        effective_recent_tools_window = (
            recent_tools_window
            if recent_tools_window is not None
            else profile_cfg.get("recent_tools_window", 2)
        )
        effective_compression_mode = (
            compression_mode
            if compression_mode is not None
            else profile_cfg.get("compression_mode")
        )
        effective_tool_selection = (
            tool_selection
            if tool_selection is not None
            else profile_cfg.get("tool_selection")
        )
        effective_tool_top_k = (
            tool_top_k
            if tool_top_k is not None
            else profile_cfg.get("tool_top_k")
        )
        effective_tool_dedup = (
            tool_dedup_old_results
            if tool_dedup_old_results is not None
            else bool(profile_cfg.get("tool_dedup_old_results", False))
        )

        self._soft_limit   = soft_token_limit
        self._hard_limit   = hard_token_limit or (
            int(soft_token_limit * 1.5) if soft_token_limit else None
        )
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
        self._cache_tools = set(cache_tools) if cache_tools else None
        self._optimization_profile = profile
        self._recent_tools_window = max(0, int(effective_recent_tools_window))
        self._compression_mode_override = _normalize_compression_mode_name(
            effective_compression_mode
        )
        self._tool_selection = _normalize_tool_selection(effective_tool_selection)
        self._tool_top_k = (
            int(effective_tool_top_k)
            if effective_tool_top_k is not None and effective_tool_top_k > 0
            else None
        )
        self._tool_min_keep = max(0, int(tool_min_keep))
        self._tool_sticky_lookback = max(0, int(tool_sticky_lookback))
        self._tool_dedup_old_results = bool(effective_tool_dedup)
        # Cache the relevance-gated tool selection within a session so the
        # provider's prompt cache (Anthropic / OpenAI) keeps prefix bytes
        # stable across consecutive `wrap_model_call` invocations in the
        # same agent.invoke loop. Without this, sticky-tool inclusion can
        # add a tool between turn N and turn N+1, mutating the tools
        # schema in the cached prefix and dropping the cache hit.
        self._tool_selection_stable = bool(tool_selection_stable)
        self._tool_selection_cache: tuple | None = None
        # Per-run snapshot of the tool-selection outcome for telemetry. Captured
        # on the first wrap_model_call of an agent run; reset after the
        # telemetry record is emitted in `_record_telemetry`.
        self._tool_selection_snapshot: dict | None = None
        # Bound checkpointed tool cache; None means caller accepts unbounded state.
        self._max_tool_cache_entries = max_tool_cache_entries
        self._token_cost_rates = token_cost_rates

        self._turn        = 0
        self._cancelled   = False
        # Use a UUID instead of `id(self)` — Python may reuse object IDs
        # after GC, so two sequential AxorMiddleware instances could end up
        # sharing a session_id and polluting any tracker that keys by it.
        import uuid as _uuid
        self._session_id  = f"lc-{_uuid.uuid4().hex[:12]}"
        self._tool_stats  = ToolStats() if track_tool_stats else None

        self._budget_tracker: Any = None
        self._budget_engine:  Any = None
        self._engines_ready = False

        # Build telemetry lazily; off-mode shows a one-time opt-in notice.
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

    def _ensure_engines(self) -> bool:
        """Init axor-core BudgetTracker on first use. False if axor-core absent."""
        if self._engines_ready:
            return True
        try:
            from axor_core.budget.tracker import BudgetTracker
            from axor_core.budget.estimator import BudgetEstimator
            from axor_core.budget.policy_engine import BudgetPolicyEngine
            self._budget_tracker = BudgetTracker()
            estimator = BudgetEstimator()
            self._budget_engine = BudgetPolicyEngine(
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
                print("[axor] axor-core not installed — budget tracking disabled")
            return False

    def _select_relevant_tools(self, tools: list, messages: list,
                                signal=None) -> list:
        """
        Drop tools whose names/descriptions don't match keywords from the
        latest user task. Always keeps:
          • tools mentioned in pending tool_calls (otherwise provider rejects),
          • tools called within the last N AI rounds (sticky),
          • at least `tool_min_keep` tools in total.

        When `signal.nature == READONLY` we additionally drop tools whose
        names match destructive verbs — "explain X" never needs `write_*`.
        """
        if (self._tool_selection != "relevance"
                or not tools
                or self._tool_top_k is None):
            return tools

        latest = _latest_human_text(messages)
        keywords = _extract_query_keywords(latest)
        if not keywords:
            return tools

        # Cache key tied to (latest user message, the available tool name
        # set). When the user hasn't sent a new message, the selection is
        # reused verbatim — this is what keeps the tools schema bytes
        # identical across consecutive `wrap_model_call` invocations and
        # lets Anthropic / OpenAI prompt cache hit on the prefix.
        cache_key: tuple | None = None
        if self._tool_selection_stable:
            tool_name_sig = tuple(_tool_attr(t, "name") for t in tools)
            cache_key = (latest, tool_name_sig)
            cached = self._tool_selection_cache
            if cached is not None and cached[0] == cache_key:
                kept_idxs = cached[1]
                # Guard: still need to honor the *current* pending set even
                # when reusing a cached decision — provider rejects orphan
                # tool_call/tool_result if a freshly-pending tool got
                # dropped from a prior cached set.
                pending = _pending_tool_call_names(messages)
                if pending:
                    pending_idxs = {
                        i for i, t in enumerate(tools)
                        if _tool_attr(t, "name") in pending
                    }
                    kept_idxs = kept_idxs | pending_idxs
                return [tools[i] for i in sorted(kept_idxs)]

        # 1. Anchors that survive any score. Sticky behavior runs on
        #    cache miss; subsequent calls within the same user turn hit
        #    the cache before reaching this point, so sticky-induced
        #    growth never mutates the cached tool list mid-turn.
        pending = _pending_tool_call_names(messages)
        sticky = _recently_called_tool_names(messages, self._tool_sticky_lookback)
        anchors = pending | sticky

        # 2. Compute task topic distribution once. Tools whose name tokens
        #    sit in the task's dominant topic get a proportional bonus
        #    inside `_score_tool_relevance` — this is what makes
        #    `security_scan` win for a security task even when the prompt
        #    uses "vulnerable" rather than "security".
        topic_strength = _compute_topic_strength(keywords)

        # 3. Score remaining tools
        scored: list[tuple[int, float, Any]] = []
        keep_idxs: set[int] = set()
        for idx, t in enumerate(tools):
            name = _tool_attr(t, "name")
            if name in anchors:
                keep_idxs.add(idx)
                continue
            score = _score_tool_relevance(
                t, keywords, topic_strength=topic_strength,
            )
            scored.append((idx, score, t))

        # 3. Optional READONLY filter: drop destructive-named tools that
        #    aren't anchored. If filtering would push us below floor we skip it.
        is_readonly = False
        try:
            from axor_core.contracts.policy import TaskNature
            is_readonly = signal is not None and signal.nature == TaskNature.READONLY
        except ImportError:
            is_readonly = False
        if is_readonly:
            kept_after_readonly = []
            for idx, score, t in scored:
                if _name_has_destructive_token(_tool_attr(t, "name")):
                    if self._verbose:
                        print(f"[axor] tool dropped (readonly task): "
                              f"{_tool_attr(t, 'name')!r}")
                    continue
                kept_after_readonly.append((idx, score, t))
            # Floor: never go below tool_min_keep + anchors total.
            min_total = max(self._tool_min_keep, len(anchors))
            if len(kept_after_readonly) + len(keep_idxs) >= min_total:
                scored = kept_after_readonly

        # 4. Top-K by score, with floor
        budget = max(0, self._tool_top_k - len(keep_idxs))
        scored.sort(key=lambda p: -p[1])
        for idx, score, _ in scored[:budget]:
            keep_idxs.add(idx)
        # Top-up to floor with the next-best (even if score is 0)
        if len(keep_idxs) < self._tool_min_keep:
            for idx, _, _ in scored[budget:]:
                keep_idxs.add(idx)
                if len(keep_idxs) >= self._tool_min_keep:
                    break

        if len(keep_idxs) >= len(tools):
            # No reduction — cache the no-op too so subsequent calls with
            # the same key skip the work entirely.
            if self._tool_selection_stable and cache_key is not None:
                self._tool_selection_cache = (cache_key, frozenset(range(len(tools))))
            return tools

        # Persist the decision under the (latest, tool_names) key so
        # follow-up calls in the same agent.invoke loop reuse it.
        if self._tool_selection_stable and cache_key is not None:
            self._tool_selection_cache = (cache_key, frozenset(keep_idxs))

        kept = [tools[i] for i in sorted(keep_idxs)]
        if self._verbose:
            dropped = [_tool_attr(t, "name") for i, t in enumerate(tools)
                       if i not in keep_idxs]
            print(f"[axor] tool relevance: kept {len(kept)}/{len(tools)}; "
                  f"dropped={dropped}")
        return kept

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

    def _merge_personality(self, request):
        """
        If personality is configured, prepend it to request.system_message and
        return a new ModelRequest via override(). Otherwise return request
        unchanged. No cache_control wrapping — that's AnthropicPromptCachingMiddleware's
        job.
        """
        if not self._personality:
            return request
        sys_msg = getattr(request, "system_message", None)
        original_text = _msg_text(sys_msg) if sys_msg is not None else ""
        merged_text = (
            f"{self._personality}\n\n{original_text}"
            if original_text else self._personality
        )
        try:
            from langchain_core.messages import SystemMessage
        except ImportError:
            return request
        if sys_msg is None:
            new_sys = SystemMessage(content=merged_text)
        else:
            try:
                new_sys = sys_msg.model_copy(update={"content": merged_text})
            except AttributeError:
                new_sys = SystemMessage(content=merged_text)
        return request.override(system_message=new_sys)

    def _resolve_execution_policy(self, messages: list):
        """
        Classify the latest user task with axor-core's HeuristicClassifier
        and pick an `ExecutionPolicy` via `PolicySelector`. The policy
        carries `compression_mode` (AGGRESSIVE / BALANCED / LIGHT) and
        `context_mode` (MINIMAL / MODERATE / BROAD), which we feed into
        the compressor and (eventually) the selector.

        Returns None if axor-core is not installed.
        """
        try:
            from axor_core.policy.selector import PolicySelector
        except ImportError:
            return None
        text = _latest_human_text(messages)
        signal = _classify_task_sync(text)
        return PolicySelector().select(signal)

    def _compress_via_axor_core(self, request):
        """
        Run the axor-core ContextCompressor on `request.messages` using
        the FragmentValue policy from `_classify_messages` AND the
        compression mode resolved by `PolicySelector` from the latest
        user task (TaskAnalyzer pipeline).

        Mode mapping examples (from PolicySelector):
          • "explain this function" → FOCUSED + READONLY → AGGRESSIVE
          • "write a test for X"    → FOCUSED + GENERATIVE → BALANCED
          • "fix the auth bug"      → FOCUSED + MUTATIVE → BALANCED
          • "rewrite the backend"   → EXPANSIVE → LIGHT

        Compressor strategy reminder:
          • PINNED fragments untouched (system, all humans, latest AI
            tool_call, recent K tool messages).
          • EPHEMERAL fragments truncated aggressively regardless of mode.
          • WORKING fragments get full pipeline by `mode` (truncate
            threshold, prose cap).
          • KNOWLEDGE fragments get dedup + error collapse only.
        """
        try:
            from axor_core.context.compressor import ContextCompressor
            from axor_core.contracts.policy import CompressionMode
        except ImportError:
            return request

        messages = list(getattr(request, "messages", []) or [])
        if not messages:
            return request

        # Explicit mode wins; otherwise policy selection falls back to balanced.
        try:
            if self._compression_mode_override is not None:
                mode = CompressionMode[self._compression_mode_override]
            else:
                policy = self._resolve_execution_policy(messages)
                mode = policy.compression_mode if policy else CompressionMode.BALANCED
        except Exception:
            mode = CompressionMode.BALANCED

        try:
            fragments = _messages_to_fragments(
                messages,
                recent_tools_window=self._recent_tools_window,
                dedup_old_results=self._tool_dedup_old_results,
            )
            compressor = ContextCompressor()
            result = compressor.compress(
                fragments,
                mode=mode,
                current_turn=len(messages) + 1,
            )
            new_messages = _fragments_to_messages(result.fragments, messages)
        except Exception as e:
            if self._verbose:
                print(f"[axor] compression failed, passing through: {e}")
            return request

        # Avoid override() when the effective message text did not change.
        if len(new_messages) == len(messages) and all(
            _msg_text(a) == _msg_text(b)
            for a, b in zip(new_messages, messages)
        ):
            return request

        if self._verbose:
            before = sum(_msg_tokens(m) for m in messages)
            after  = sum(_msg_tokens(m) for m in new_messages)
            print(f"[axor] compress: {before}→{after} tokens "
                  f"({len(messages)}→{len(new_messages)} msgs, "
                  f"mode={mode.name}, "
                  f"strategies: {result.strategies_applied})")

        return request.override(messages=new_messages)

    def _enforce_hard_limit(self, request) -> None:
        """
        Raise `BudgetExceededError` if the upcoming call would exceed the
        hard token limit. Estimate-based gate; no state mutation.
        """
        if not self._engines_ready or not self._hard_limit:
            return
        total_spent = self._budget_tracker.total_tokens()
        messages = list(getattr(request, "messages", []) or [])
        sys_msg = getattr(request, "system_message", None)
        if sys_msg is not None:
            messages = [sys_msg, *messages]
        projected = sum(_msg_tokens(m) for m in messages)

        if self._soft_limit and total_spent > self._soft_limit and self._verbose:
            print(f"[axor] soft limit warning: "
                  f"{total_spent}/{self._soft_limit} tokens")

        if total_spent + projected > self._hard_limit:
            if self._verbose:
                print(f"[axor] hard budget stop: {total_spent} spent + "
                      f"{projected} projected > {self._hard_limit}")
            from axor_core.errors import BudgetExceededError
            raise BudgetExceededError(
                spent=total_spent,
                projected=projected,
                limit=self._hard_limit,
            )

    def _record_usage_from_response(self, response) -> None:
        """
        Record provider-counted input, output, and prompt-cache tokens.

        Anthropic reports cache_creation_input_tokens and
        cache_read_input_tokens separately from input_tokens. Recording all
        counters here keeps Axor's budget total aligned with provider usage
        volume; pre-call estimation never touches the tracker.
        """
        if not self._engines_ready:
            return
        usages = []
        direct_usage = getattr(response, "usage_metadata", None)
        if direct_usage:
            usages.append(direct_usage)
        for msg in getattr(response, "result", []) or []:
            msg_usage = getattr(msg, "usage_metadata", None)
            if msg_usage:
                usages.append(msg_usage)
        if not usages:
            return

        in_t = 0
        out_t = 0
        cache_creation_t = 0
        cache_read_t = 0
        for usage in usages:
            if isinstance(usage, dict):
                in_t  += usage.get("input_tokens", 0) or 0
                out_t += usage.get("output_tokens", 0) or 0
                cache_creation_t += (
                    usage.get("cache_creation_input_tokens", 0) or 0
                )
                cache_read_t += usage.get("cache_read_input_tokens", 0) or 0
            else:
                in_t  += getattr(usage, "input_tokens", 0) or 0
                out_t += getattr(usage, "output_tokens", 0) or 0
                cache_creation_t += (
                    getattr(usage, "cache_creation_input_tokens", 0) or 0
                )
                cache_read_t += (
                    getattr(usage, "cache_read_input_tokens", 0) or 0
                )
        self._budget_tracker.record(
            node_id=self._session_id,
            input_tokens=in_t,
            output_tokens=out_t,
            cache_creation_input_tokens=cache_creation_t,
            cache_read_input_tokens=cache_read_t,
        )
        if self._verbose:
            total = self._budget_tracker.total_tokens()
            cache_part = ""
            if cache_creation_t or cache_read_t:
                cache_part = (
                    f" / +{cache_creation_t} cache_write"
                    f" / +{cache_read_t} cache_read"
                )
            print(f"[axor] usage: +{in_t} in / +{out_t} out{cache_part} "
                  f"tokens (total: {total})")

    def _execute_tool_governed(self, tool_name: str, request, handler: Callable) -> Any:
        attempts = self._tool_max_retries + 1
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
                    time.sleep(self._tool_retry_delay)

        latency_ms = (time.monotonic() - t0) * 1000
        if self._tool_stats:
            self._tool_stats.record_call(tool_name, latency_ms)
            self._tool_stats.record_error(tool_name)

        if self._tool_error_handler is not None:
            msg = self._tool_error_handler(tool_name, last_error)
            if self._verbose:
                print(f"[axor] → {tool_name!r} handled error")
            return _HandledError(msg) if isinstance(msg, str) else msg

        raise last_error

    async def _aexecute_tool_governed(self, tool_name: str, request, handler: Callable) -> Any:
        attempts = self._tool_max_retries + 1
        last_error: Exception | None = None
        t0 = time.monotonic()

        for attempt in range(1, attempts + 1):
            try:
                result = await handler(request)
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
                    await asyncio.sleep(self._tool_retry_delay)

        latency_ms = (time.monotonic() - t0) * 1000
        if self._tool_stats:
            self._tool_stats.record_call(tool_name, latency_ms)
            self._tool_stats.record_error(tool_name)

        if self._tool_error_handler is not None:
            msg = self._tool_error_handler(tool_name, last_error)
            if self._verbose:
                print(f"[axor] → {tool_name!r} handled error")
            return _HandledError(msg) if isinstance(msg, str) else msg

        raise last_error

    def _prepare_model_request(self, request):
        self._ensure_engines()
        self._turn += 1

        tools = getattr(request, "tools", None) or []
        offered_count = len(tools)
        filtered = self._filter_tools(tools)
        dropped_denied = offered_count - len(filtered)
        if dropped_denied:
            request = request.override(tools=filtered)

        request = self._merge_personality(request)
        request = self._compress_via_axor_core(request)

        # Tool relevance gating runs after compression so anchors derived
        # from `messages` see compression's view of pending tool_calls.
        kept_count = len(filtered)
        dropped_relevance = 0
        if self._tool_selection == "relevance":
            current_tools = getattr(request, "tools", None) or []
            messages = list(getattr(request, "messages", []) or [])
            signal = None
            try:
                signal = _classify_task_sync(_latest_human_text(messages))
            except Exception:
                signal = None
            relevant = self._select_relevant_tools(current_tools, messages, signal)
            dropped_relevance = len(current_tools) - len(relevant)
            kept_count = len(relevant)
            if dropped_relevance:
                request = request.override(tools=relevant)

        # Snapshot the tool-selection outcome for telemetry. We overwrite
        # each turn rather than guarding on first-call because relevance
        # caching + static allow/deny lists make the numbers stable across
        # turns within a run, and overwrite is robust to runs that share a
        # middleware instance without telemetry-driven reset.
        if offered_count > 0:
            self._tool_selection_snapshot = {
                "mode":              self._tool_selection or "none",
                "offered":           offered_count,
                "kept":              kept_count,
                "dropped_relevance": dropped_relevance,
                "dropped_denied":    dropped_denied,
            }

        self._enforce_hard_limit(request)
        return request

    @property
    def total_tokens_spent(self) -> int:
        return self._budget_tracker.total_tokens() if self._engines_ready else 0

    def cost_summary(self) -> dict | None:
        if not self._engines_ready or self._token_cost_rates is None:
            return None
        return self._budget_tracker.cost_summary(self._token_cost_rates)

    @property
    def turns(self) -> int:
        return self._turn

    @property
    def tool_stats(self) -> ToolStats | None:
        return self._tool_stats

    def cancel(self, reason: str = "user_abort") -> None:
        """Signal cancellation. Safe to call from sync, async, or signal handler.

        After cancel(), the next `wrap_model_call` / `wrap_tool_call`
        invocation raises `AxorCancelledError` instead of dispatching to
        the handler — terminating the agent run early.

        Idempotent — repeated calls keep the original reason.
        """
        if not self._cancelled:
            self._cancelled = True
            self._cancel_reason = reason

    def is_cancelled(self) -> bool:
        return self._cancelled

    def _check_cancel(self, where: str) -> None:
        """Raise AxorCancelledError if cancellation has been requested.

        Called at every model-call and tool-call boundary so a `cancel()`
        in another thread or coroutine takes effect within one event step.
        """
        if self._cancelled:
            raise AxorCancelledError(
                f"axor: cancelled at {where} (reason={getattr(self, '_cancel_reason', 'user_abort')})"
            )

    def reset(self) -> None:
        self._turn      = 0
        self._cancelled = False
        self._cancel_reason = "user_abort"
        if self._tool_stats:
            self._tool_stats.reset()
        self._budget_tracker = None
        self._budget_engine  = None
        self._engines_ready  = False
        self._tool_selection_cache = None
        self._tool_selection_snapshot = None

def _make_axor_middleware_class():
    from langchain.agents.middleware import (
        AgentMiddleware,
        AgentState,
        ModelRequest,
        ModelResponse,
        ToolCallRequest,
    )
    from langchain_core.messages import ToolMessage
    from langgraph.runtime import Runtime
    from langgraph.types import Command

    class AxorMiddlewareImpl(_AxorGovernanceCore, AgentMiddleware):

        # Adds `tool_result_cache` to LangGraph state for checkpointed memoization.
        state_schema = AxorState

        def __init__(self, **kwargs) -> None:
            _AxorGovernanceCore.__init__(self, **kwargs)
            AgentMiddleware.__init__(self)

        def wrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], ModelResponse],
        ) -> ModelResponse:
            """
            Single-hook pipeline:
              1. Bail out if cancellation was requested.
              2. Filter tools by allow/deny lists.
              3. Merge `personality` into system_message.
              4. Compress messages via axor-core ContextCompressor with
                 FragmentValue-aware semantics (PINNED untouched, etc.)
                 and CompressionMode picked from the latest task.
              5. Hard-budget gate (raises BudgetExceededError when over).
              6. Hand off to handler.
              7. Record provider-counted input + output tokens.
            """
            self._check_cancel("wrap_model_call")
            request = self._prepare_model_request(request)
            self._check_cancel("wrap_model_call(post-prepare)")
            response = handler(request)
            self._record_usage_from_response(response)
            return response

        async def awrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], Any],
        ) -> ModelResponse:
            self._check_cancel("awrap_model_call")
            request = self._prepare_model_request(request)
            self._check_cancel("awrap_model_call(post-prepare)")
            response = await handler(request)
            self._record_usage_from_response(response)
            return response

        def wrap_tool_call(
            self,
            request: ToolCallRequest,
            handler: Callable[[ToolCallRequest], Any],
        ) -> Any:
            """
            Tool result memoization via state["tool_result_cache"].

            Same (tool_name, args) → same cached content. Across
            invocations under the same `thread_id`, the cache survives
            via LangGraph checkpointing (because we declared state_schema
            = AxorState).

            Errors are NOT cached — a transient failure shouldn't replay
            forever; the next call should retry.

            On cache miss, we return a Command bundling (a) the new
            ToolMessage and (b) the state update with the cache entry.
            On cache hit, we return a plain ToolMessage; no state update
            needed (cache already contains this entry).
            """
            self._check_cancel("wrap_tool_call")
            tc        = request.tool_call if hasattr(request, "tool_call") else {}
            tool_name = _tool_name(tc)
            tc_id     = _tool_call_id(tc)
            args      = (tc.get("args") if isinstance(tc, dict)
                         else getattr(tc, "args", {}))

            if self._verbose:
                print(f"[axor] → tool {tool_name!r} args={args}")

            state = getattr(request, "state", None) or {}
            existing_cache = (
                state.get("tool_result_cache") if isinstance(state, dict)
                else getattr(state, "tool_result_cache", None)
            ) or {}

            cacheable = _is_tool_call_cacheable(tool_name, self._cache_tools)
            cache_key = _hash_tool_call(tool_name, args) if cacheable else ""

            if cacheable and cache_key in existing_cache:
                if self._verbose:
                    print(f"[axor] tool cache hit: {tool_name!r}")
                if self._tool_stats:
                    self._tool_stats.record_cache_hit(tool_name)
                # Cache hits still add content to the next prompt budget.
                cached_content = existing_cache[cache_key]
                if self._engines_ready:
                    self._budget_tracker.record(
                        node_id=self._session_id,
                        input_tokens=0, output_tokens=0,
                        tool_tokens=max(len(cached_content) // 4, 1),
                    )
                return ToolMessage(
                    content=cached_content,
                    tool_call_id=tc_id or "",
                )

            result = self._execute_tool_governed(tool_name, request, handler)

            if isinstance(result, _HandledError):
                return ToolMessage(content=str(result), tool_call_id=tc_id or "")

            if isinstance(result, str):
                out_text = result
                msg = ToolMessage(content=result, tool_call_id=tc_id or "")
            elif isinstance(result, ToolMessage):
                content = result.content
                out_text = content if isinstance(content, str) else str(content or "")
                msg = result
            else:
                return result

            if self._engines_ready:
                self._budget_tracker.record(
                    node_id=self._session_id,
                    input_tokens=0, output_tokens=0,
                    tool_tokens=max(len(out_text) // 4, 1),
                )

            if not cacheable:
                return msg

            new_cache = {**existing_cache, cache_key: out_text}

            # FIFO eviction keeps checkpointed state bounded.
            cap = self._max_tool_cache_entries
            if cap is not None and len(new_cache) > cap:
                for stale_key in list(new_cache.keys())[: len(new_cache) - cap]:
                    del new_cache[stale_key]

            return Command(update={
                "messages": [msg],
                "tool_result_cache": new_cache,
            })

        async def awrap_tool_call(
            self,
            request: ToolCallRequest,
            handler: Callable[[ToolCallRequest], Any],
        ) -> Any:
            tc        = request.tool_call if hasattr(request, "tool_call") else {}
            tool_name = _tool_name(tc)
            tc_id     = _tool_call_id(tc)
            args      = (tc.get("args") if isinstance(tc, dict)
                         else getattr(tc, "args", {}))

            if self._verbose:
                print(f"[axor] → tool {tool_name!r} args={args}")

            state = getattr(request, "state", None) or {}
            existing_cache = (
                state.get("tool_result_cache") if isinstance(state, dict)
                else getattr(state, "tool_result_cache", None)
            ) or {}

            cacheable = _is_tool_call_cacheable(tool_name, self._cache_tools)
            cache_key = _hash_tool_call(tool_name, args) if cacheable else ""

            if cacheable and cache_key in existing_cache:
                if self._verbose:
                    print(f"[axor] tool cache hit: {tool_name!r}")
                if self._tool_stats:
                    self._tool_stats.record_cache_hit(tool_name)
                cached_content = existing_cache[cache_key]
                if self._engines_ready:
                    self._budget_tracker.record(
                        node_id=self._session_id,
                        input_tokens=0, output_tokens=0,
                        tool_tokens=max(len(cached_content) // 4, 1),
                    )
                return ToolMessage(
                    content=cached_content,
                    tool_call_id=tc_id or "",
                )

            result = await self._aexecute_tool_governed(tool_name, request, handler)

            if isinstance(result, _HandledError):
                return ToolMessage(content=str(result), tool_call_id=tc_id or "")

            if isinstance(result, str):
                out_text = result
                msg = ToolMessage(content=result, tool_call_id=tc_id or "")
            elif isinstance(result, ToolMessage):
                content = result.content
                out_text = content if isinstance(content, str) else str(content or "")
                msg = result
            else:
                return result

            if self._engines_ready:
                self._budget_tracker.record(
                    node_id=self._session_id,
                    input_tokens=0, output_tokens=0,
                    tool_tokens=max(len(out_text) // 4, 1),
                )

            if not cacheable:
                return msg

            new_cache = {**existing_cache, cache_key: out_text}
            cap = self._max_tool_cache_entries
            if cap is not None and len(new_cache) > cap:
                for stale_key in list(new_cache.keys())[: len(new_cache) - cap]:
                    del new_cache[stale_key]

            return Command(update={
                "messages": [msg],
                "tool_result_cache": new_cache,
            })

        def after_agent(
            self,
            state: AgentState,
            runtime: Runtime,
        ) -> dict[str, Any] | None:
            if self._verbose and self._tool_stats:
                print(f"[axor] {self._tool_stats.summary()}")
            if self._verbose and self._engines_ready:
                print(f"[axor] session total tokens: {self._budget_tracker.total_tokens()}")
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
                # Memory persistence is best-effort — do NOT crash the agent
                # run on a provider failure. But the previous behaviour
                # (silent unless verbose) hid recurring failures from
                # operators. Always log via the package logger AND surface
                # to stderr so a degraded provider isn't invisible. Verbose
                # callers also see it in their stdout stream.
                _log.warning("memory save failed: %s", e, exc_info=True)
                print(f"[axor] memory save failed: {e}", file=sys.stderr)
            return None

        async def _record_telemetry(self, state) -> None:
            """Forward one AnonymizedTraceRecord per agent run. Silent no-op when off."""
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
                tool_selection = self._tool_selection_snapshot
                await self._telemetry.record_decision(
                    raw_input=human,
                    signal=signal,
                    classifier_used=event.classifier,
                    confidence=float(event.confidence),
                    tokens_spent=int(tokens),
                    policy_adjusted=False,
                    tool_selection=tool_selection,
                )
                # Reset only after a successful enqueue — if record_decision
                # failed earlier, the caller's `except` already swallowed it
                # and the next run will report fresh stats.
                self._tool_selection_snapshot = None
            except Exception as e:
                # Telemetry must never break the host; log via package logger
                # and (if verbose) surface to stdout. Stderr fallback skipped —
                # opt-in telemetry failing shouldn't pollute non-verbose output.
                _log.warning("telemetry record failed: %s", e, exc_info=True)
                if self._verbose:
                    print(f"[axor] telemetry record failed: {e}")

    AxorMiddlewareImpl.__name__ = "AxorMiddleware"
    return AxorMiddlewareImpl

class AxorMiddleware(_AxorGovernanceCore):
    """
    Governance middleware for LangChain 1.0 agents.

    What it does (in `wrap_model_call` order): filter tools, merge optional
    personality into system_message, compress messages via axor-core's
    FragmentValue-aware `ContextCompressor` (mode picked by `PolicySelector`
    from the latest user task), gate against hard token limit, hand off
    to the underlying handler, then record provider-counted token usage.

    `wrap_tool_call` adds optional tool-result memoization for tools listed
    in `cache_tools`. The cache lives in `tool_result_cache` on the custom
    `AxorState` and survives across `agent.invoke()` calls under the same
    `thread_id` via LangGraph checkpointing.

    Provider-side caching is delegated to
    `langchain_anthropic.AnthropicPromptCachingMiddleware`; compose them
    side by side. AxorMiddleware does not add `cache_control` markers.

    Parameters
    ----------
    soft_token_limit : int | None
        Advisory threshold — logs a warning when crossed.
    hard_token_limit : int | None
        Hard stop — raises `axor_core.errors.BudgetExceededError` when the
        next call's projected input would exceed this. Defaults to
        soft_token_limit * 1.5 when only soft is set.
    allowed_tools : list[str] | None
        Whitelist — only these tools reach the model.
    denied_tools : list[str] | None
        Blacklist — these tools are stripped from every model call.
    personality : str | None
        Prepended to system_message verbatim.
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
    telemetry : str | None
        "off" | "local" | "remote". Resolves from AXOR_TELEMETRY env if None.
    cache_tools : list[str] | None
        Tool names whose results may be memoized. Defaults to None because
        LangChain tools may have side effects or return time-sensitive data.
    token_cost_rates : axor_core.budget.TokenCostRates | None
        Optional provider/model pricing. When set, `cost_summary()` returns
        estimated money cost with prompt-cache write/read multipliers.
    optimization_profile : str | None
        Named compression preset. "cautious" favors quality and keeps the last
        two tool results verbatim without tool relevance filtering.
        "aggressive" favors cost with aggressive compression and relevance-based
        tool schema trimming. Use `compression_mode="balanced"` as an explicit
        override when you want moderate compression without a named profile.
    recent_tools_window : int
        Number of most recent ToolMessages kept verbatim before compression.
        Lower values can save more tokens on large-output agents, but may
        increase the risk of repeated tool calls. Defaults to the profile value
        or 2 when no profile is set.
    compression_mode : str | None
        Optional override for policy-selected compression. Accepts "auto",
        "aggressive"/"minimal", "balanced"/"moderate", or "light"/"broad".

    Examples
    --------
    Basic::

        axor = AxorMiddleware(soft_token_limit=80_000, verbose=True)
        agent = create_agent("anthropic:claude-sonnet-4-6", tools=tools, middleware=[axor])

    Compose with Anthropic prompt caching (recommended)::

        from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware
        agent = create_agent(
            "anthropic:claude-sonnet-4-6",
            tools=tools,
            middleware=[
                AnthropicPromptCachingMiddleware(),
                AxorMiddleware(soft_token_limit=80_000),
            ],
        )
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
            cls._impl_class.__init__(instance, **kwargs)
            return instance
        return object.__new__(cls)

    def __init__(
        self,
        soft_token_limit: int | None = None,
        hard_token_limit: int | None = None,
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
        cache_tools: list[str] | None = None,
        max_tool_cache_entries: int | None = 100,
        recent_tools_window: int | None = None,
        compression_mode: str | None = None,
        optimization_profile: str | None = None,
        tool_selection: str | None = None,
        tool_top_k: int | None = None,
        tool_min_keep: int = 3,
        tool_sticky_lookback: int = 4,
        tool_dedup_old_results: bool | None = None,
        tool_selection_stable: bool = True,
        token_cost_rates: Any | None = None,
    ) -> None:
        if getattr(self, "_engines_ready", None) is not None:
            return
        _AxorGovernanceCore.__init__(
            self,
            soft_token_limit=soft_token_limit,
            hard_token_limit=hard_token_limit,
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
            cache_tools=cache_tools,
            max_tool_cache_entries=max_tool_cache_entries,
            recent_tools_window=recent_tools_window,
            compression_mode=compression_mode,
            optimization_profile=optimization_profile,
            tool_selection=tool_selection,
            tool_top_k=tool_top_k,
            tool_min_keep=tool_min_keep,
            tool_sticky_lookback=tool_sticky_lookback,
            tool_dedup_old_results=tool_dedup_old_results,
            tool_selection_stable=tool_selection_stable,
            token_cost_rates=token_cost_rates,
        )
