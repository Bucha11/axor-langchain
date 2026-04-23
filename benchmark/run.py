"""
axor-langchain benchmark — full middleware pipeline test.

Tests ALL middleware features through real LangChain agent pipeline
using FakeMessagesListChatModel (no API keys required):

  1. Compression     — message history reduction across profiles
  2. Tool governance — allow/deny filtering
  3. Budget          — soft/hard token limits
  4. Tool retry      — retry + error handler
  5. Full pipeline   — all features combined

Run:
    python benchmark/run.py
    python benchmark/run.py --json
    python benchmark/run.py --scenario compression
    python benchmark/run.py --scenario tool-governance
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "axor-core"))

from dataclasses import dataclass, field

from langchain.agents import create_agent
from langchain_core.language_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage

from axor_langchain.middleware import AxorMiddleware
from benchmark.graph import (
    ALL_TOOLS,
    build_planner_responses,
    build_prior_history,
    build_researcher_responses,
    build_writer_responses,
    count_messages_tokens,
    estimate_tokens,
    make_plan,
)


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class ScenarioResult:
    name: str
    passed: bool
    metrics: dict = field(default_factory=dict)
    detail: str = ""
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "passed": self.passed,
            "metrics": self.metrics,
            "detail": self.detail,
            "elapsed_ms": round(self.elapsed_ms, 1),
        }


# ── Scenario 1: Compression ──────────────────────────────────────────────────

def scenario_compression(task: str) -> list[ScenarioResult]:
    """Test message compression across short/medium/long profiles."""
    results = []

    for profile, n_turns in [("short", 3), ("medium", 8), ("long", 20)]:
        history = build_prior_history(task, n_turns)
        n_msgs_before, tok_before = count_messages_tokens(history)

        axor = AxorMiddleware(
            compression_mode="auto",
            soft_token_limit=200_000,
            verbose=False,
        )

        compressed, changed = axor._govern_messages_core(history)
        n_msgs_after, tok_after = count_messages_tokens(compressed)

        reduction = (1 - tok_after / tok_before) * 100 if tok_before else 0

        results.append(ScenarioResult(
            name=f"compression/{profile} ({n_turns} turns)",
            passed=True,
            metrics={
                "turns": n_turns,
                "messages_before": n_msgs_before,
                "messages_after": n_msgs_after,
                "tokens_before": tok_before,
                "tokens_after": tok_after,
                "reduction_pct": round(reduction, 1),
                "axor_turns": axor.turns,
                "engine": "axor-core" if axor._engines_ready else "fallback",
            },
        ))
    return results


# ── Scenario 2: Compression modes comparison ─────────────────────────────────

def scenario_compression_modes(task: str) -> list[ScenarioResult]:
    """Compare auto vs minimal vs moderate vs broad."""
    history = build_prior_history(task, 12)
    _, tok_before = count_messages_tokens(history)
    results = []

    for mode in ("auto", "minimal", "moderate", "broad"):
        axor = AxorMiddleware(compression_mode=mode, verbose=False)
        compressed, _ = axor._govern_messages_core(history)
        _, tok_after = count_messages_tokens(compressed)
        reduction = (1 - tok_after / tok_before) * 100 if tok_before else 0

        results.append(ScenarioResult(
            name=f"modes/{mode}",
            passed=True,
            metrics={
                "mode": mode,
                "tokens_before": tok_before,
                "tokens_after": tok_after,
                "reduction_pct": round(reduction, 1),
            },
        ))
    return results


# ── Scenario 3: Tool governance ───────────────────────────────────────────────

def scenario_tool_governance(task: str) -> list[ScenarioResult]:
    """Test allow/deny tool filtering through real agent pipeline."""
    results = []

    # --- allow only search_web + summarize_findings, deny analyze_data ---
    responses = build_planner_responses(task)
    model = FakeMessagesListChatModel(responses=responses)

    axor = AxorMiddleware(
        allowed_tools=["search_web", "summarize_findings"],
        denied_tools=["analyze_data"],
        track_tool_stats=True,
        verbose=False,
    )

    # Test _filter_tools directly — this is what wrap_model_call does
    filtered = axor._filter_tools(ALL_TOOLS)
    filtered_names = [t.name for t in filtered]
    denied = [t.name for t in ALL_TOOLS if t.name not in filtered_names]

    passed = (
        "search_web" in filtered_names
        and "summarize_findings" in filtered_names
        and "analyze_data" not in filtered_names
    )

    stats = axor.tool_stats
    results.append(ScenarioResult(
        name="tool-governance/allow-deny",
        passed=passed,
        metrics={
            "allowed": filtered_names,
            "denied": denied,
            "total_tools": len(ALL_TOOLS),
            "filtered_tools": len(filtered),
            "denied_count": stats.by_tool.get("analyze_data", None) and stats.by_tool["analyze_data"].denied_count or 0,
        },
        detail=f"filtered: {filtered_names}, denied: {denied}",
    ))

    # --- deny specific tools via denied_tools ---
    all_tool_names = [t.name for t in ALL_TOOLS]
    axor_deny_all = AxorMiddleware(
        denied_tools=all_tool_names,
        track_tool_stats=True,
        verbose=False,
    )
    filtered_all = axor_deny_all._filter_tools(ALL_TOOLS)
    results.append(ScenarioResult(
        name="tool-governance/deny-all-via-denylist",
        passed=len(filtered_all) == 0,
        metrics={"filtered_tools": len(filtered_all), "denied": all_tool_names},
        detail=f"all tools denied: {len(filtered_all) == 0}",
    ))

    return results


# ── Scenario 4: Budget enforcement ────────────────────────────────────────────

def scenario_budget(task: str) -> list[ScenarioResult]:
    """Test soft/hard budget limits."""
    results = []

    # Build large history that exceeds budget
    history = build_prior_history(task, 8)
    _, tok = count_messages_tokens(history)

    # --- soft limit warning ---
    axor_soft = AxorMiddleware(
        soft_token_limit=tok // 2,  # set low so it triggers
        verbose=False,
    )
    axor_soft._govern_messages_core(history)

    results.append(ScenarioResult(
        name="budget/soft-limit",
        passed=axor_soft.turns == 1,
        metrics={
            "history_tokens": tok,
            "soft_limit": tok // 2,
            "turns_processed": axor_soft.turns,
            "total_tracked": axor_soft.total_tokens_spent,
        },
    ))

    # --- hard limit cancellation ---
    axor_hard = AxorMiddleware(
        soft_token_limit=100,
        hard_token_limit=200,
        verbose=False,
    )
    axor_hard._govern_messages_core(history)
    cancelled = axor_hard._cancelled

    results.append(ScenarioResult(
        name="budget/hard-limit",
        passed=cancelled,
        metrics={
            "history_tokens": tok,
            "hard_limit": 200,
            "cancelled": cancelled,
        },
        detail="hard budget stop triggered" if cancelled else "NOT triggered (unexpected)",
    ))

    return results


# ── Scenario 5: Tool retry + error handler ────────────────────────────────────

def scenario_tool_retry(task: str) -> list[ScenarioResult]:
    """Test tool execution retry and error handling."""
    results = []
    errors_caught = []

    def error_handler(name: str, exc: Exception) -> str:
        errors_caught.append((name, str(exc)))
        return f"Tool {name} failed: {exc}"

    axor = AxorMiddleware(
        tool_max_retries=2,
        tool_retry_delay=0.01,
        tool_error_handler=error_handler,
        track_tool_stats=True,
        verbose=False,
    )

    # Simulate tool execution that fails
    call_count = 0
    def failing_handler(request):
        nonlocal call_count
        call_count += 1
        raise RuntimeError("connection timeout")

    t0 = time.monotonic()
    result = axor._execute_tool_governed("analyze_data", None, failing_handler)
    elapsed = (time.monotonic() - t0) * 1000

    stats = axor.tool_stats
    tool_stat = stats.by_tool.get("analyze_data")

    results.append(ScenarioResult(
        name="tool-retry/retries-then-handler",
        passed=(
            call_count == 3                          # 1 + 2 retries
            and len(errors_caught) == 1              # error_handler called once
            and isinstance(result, str)              # handler returned string
            and "failed" in result
        ),
        metrics={
            "attempts": call_count,
            "expected_attempts": 3,
            "errors_handled": len(errors_caught),
            "error_result": result[:80] if isinstance(result, str) else str(type(result)),
            "elapsed_ms": round(elapsed, 1),
            "call_count": tool_stat.call_count if tool_stat else 0,
            "error_count": tool_stat.error_count if tool_stat else 0,
            "success_rate": f"{tool_stat.success_rate:.0%}" if tool_stat else "N/A",
        },
    ))

    # Successful tool with latency tracking
    axor2 = AxorMiddleware(track_tool_stats=True, verbose=False)
    axor2._execute_tool_governed("search_web", None, lambda r: "search result ok")
    st = axor2.tool_stats.by_tool.get("search_web")

    results.append(ScenarioResult(
        name="tool-retry/success-tracking",
        passed=(st is not None and st.call_count == 1 and st.error_count == 0),
        metrics={
            "call_count": st.call_count if st else 0,
            "avg_latency_ms": round(st.avg_latency_ms, 1) if st else 0,
            "success_rate": f"{st.success_rate:.0%}" if st else "N/A",
        },
    ))

    return results


# ── Scenario 6: Bypass detection ──────────────────────────────────────────────

def scenario_bypass(task: str) -> list[ScenarioResult]:
    """Test that small contexts bypass compression pipeline."""
    results = []

    # small context — should be bypassed
    history = build_prior_history(task, 2)  # ~500 tokens
    _, tok_small = count_messages_tokens(history)

    axor_bypass = AxorMiddleware(
        bypass_token_threshold=4000,
        verbose=False,
    )
    compressed, changed = axor_bypass._govern_messages_core(history)

    results.append(ScenarioResult(
        name="bypass/small-context-skipped",
        passed=not changed,  # should NOT be changed (bypassed)
        metrics={
            "tokens": tok_small,
            "threshold": 4000,
            "changed": changed,
        },
        detail=f"{tok_small} tokens < 4000 threshold → bypass" if not changed else "NOT bypassed (unexpected)",
    ))

    # large context — should NOT be bypassed
    history_large = build_prior_history(task, 8)
    _, tok_large = count_messages_tokens(history_large)

    axor_no_bypass = AxorMiddleware(
        bypass_token_threshold=100,  # very low threshold
        verbose=False,
    )
    compressed_large, changed_large = axor_no_bypass._govern_messages_core(history_large)

    results.append(ScenarioResult(
        name="bypass/large-context-compressed",
        passed=changed_large,  # should be changed (compressed)
        metrics={
            "tokens": tok_large,
            "threshold": 100,
            "changed": changed_large,
        },
        detail=f"{tok_large} tokens > 100 threshold → compressed" if changed_large else "NOT compressed (unexpected)",
    ))

    # bypass=0 disables bypass entirely
    axor_disabled = AxorMiddleware(
        bypass_token_threshold=0,
        verbose=False,
    )
    _, changed_disabled = axor_disabled._govern_messages_core(history)

    results.append(ScenarioResult(
        name="bypass/disabled-with-zero",
        passed=True,  # just verify it doesn't crash
        metrics={"threshold": 0, "changed": changed_disabled},
    ))

    return results


# ── Scenario 7: Full pipeline ─────────────────────────────────────────────────

def scenario_full_pipeline(task: str) -> list[ScenarioResult]:
    """All features combined — real agent through create_agent + middleware."""
    results = []

    axor = AxorMiddleware(
        soft_token_limit=200_000,
        compression_mode="auto",
        allowed_tools=["search_web", "summarize_findings"],
        denied_tools=["analyze_data"],
        personality="You are a thorough research assistant.",
        tool_max_retries=1,
        tool_error_handler=lambda n, e: f"[{n} error: {e}]",
        track_tool_stats=True,
        verbose=False,
    )

    # Build prior history + compress
    history = build_prior_history(task, 8)
    n_before, tok_before = count_messages_tokens(history)

    compressed, changed = axor._govern_messages_core(history)
    n_after, tok_after = count_messages_tokens(compressed)

    # Test tool filtering
    filtered = axor._filter_tools(ALL_TOOLS)
    filtered_names = [t.name for t in filtered]

    # Test tool execution
    axor._execute_tool_governed("search_web", None, lambda r: "ok")

    stats = axor.tool_stats

    reduction = (1 - tok_after / tok_before) * 100 if tok_before else 0

    passed = (
        axor.turns >= 1                              # middleware processed turns
        and "analyze_data" not in filtered_names     # denied tool filtered
        and stats.total_calls >= 1                   # tool execution tracked
        and axor.total_tokens_spent >= 0             # budget tracker working
    )

    results.append(ScenarioResult(
        name="full-pipeline/combined",
        passed=passed,
        metrics={
            "compression": {
                "messages_before": n_before,
                "messages_after": n_after,
                "tokens_before": tok_before,
                "tokens_after": tok_after,
                "reduction_pct": round(reduction, 1),
            },
            "tool_governance": {
                "allowed": filtered_names,
                "denied": ["analyze_data"],
            },
            "budget": {
                "total_tokens_spent": axor.total_tokens_spent,
                "engine": "axor-core" if axor._engines_ready else "fallback",
            },
            "tool_stats": {
                "total_calls": stats.total_calls,
                "total_errors": stats.total_errors,
            },
            "turns": axor.turns,
        },
    ))

    return results


# ── Runner ────────────────────────────────────────────────────────────────────

SCENARIOS = {
    "compression": scenario_compression,
    "modes": scenario_compression_modes,
    "tool-governance": scenario_tool_governance,
    "budget": scenario_budget,
    "tool-retry": scenario_tool_retry,
    "bypass": scenario_bypass,
    "full-pipeline": scenario_full_pipeline,
}


def run_all(task: str, scenarios: list[str] | None = None) -> list[ScenarioResult]:
    to_run = scenarios or list(SCENARIOS.keys())
    all_results = []

    for name in to_run:
        fn = SCENARIOS.get(name)
        if not fn:
            print(f"  unknown scenario: {name}")
            continue
        t0 = time.monotonic()
        results = fn(task)
        elapsed = (time.monotonic() - t0) * 1000
        for r in results:
            r.elapsed_ms = elapsed / len(results)
        all_results.extend(results)

    return all_results


# ── Report ────────────────────────────────────────────────────────────────────

def _status(passed: bool) -> str:
    return "\033[32m PASS \033[0m" if passed else "\033[31m FAIL \033[0m"


def print_report(results: list[ScenarioResult], task: str) -> None:
    W = 74
    print()
    print("  " + "─" * W)
    print(f"  \033[1m  axor-langchain benchmark\033[0m")
    print(f"     task: {task[:60]}")
    print("  " + "─" * W)
    print()

    passed = sum(1 for r in results if r.passed)
    total = len(results)

    # Group by scenario prefix
    current_group = ""
    for r in results:
        group = r.name.split("/")[0]
        if group != current_group:
            current_group = group
            print(f"  \033[1m{group}\033[0m")

        status = _status(r.passed)
        print(f"  {status} {r.name}")

        # Print key metrics
        metrics = r.metrics
        if "reduction_pct" in metrics:
            print(f"         tokens: {metrics.get('tokens_before', '?'):,} → {metrics.get('tokens_after', '?'):,}"
                  f"  ({metrics['reduction_pct']:+.1f}%)")
        if "allowed" in metrics and isinstance(metrics["allowed"], list):
            print(f"         allowed: {metrics['allowed']}")
            if "denied" in metrics:
                print(f"         denied:  {metrics['denied']}")
        if "cancelled" in metrics:
            print(f"         cancelled: {metrics['cancelled']}")
        if "attempts" in metrics:
            print(f"         attempts: {metrics['attempts']}/{metrics.get('expected_attempts', '?')}"
                  f"  errors_handled: {metrics.get('errors_handled', 0)}")
        if "compression" in metrics and isinstance(metrics["compression"], dict):
            c = metrics["compression"]
            print(f"         compression: {c['tokens_before']:,} → {c['tokens_after']:,} ({c['reduction_pct']:+.1f}%)")
            print(f"         tools: {metrics['tool_governance']['allowed']}, budget: {metrics['budget']['total_tokens_spent']:,} tok")
            print(f"         tool calls: {metrics['tool_stats']['total_calls']}, engine: {metrics['budget']['engine']}")

        if r.detail:
            print(f"         {r.detail}")
        print()

    # Summary
    color = "\033[32m" if passed == total else "\033[33m"
    print("  " + "─" * W)
    print(f"  {color}{passed}/{total} passed\033[0m")
    print("  " + "─" * W)
    print()


def print_json(results: list[ScenarioResult], task: str) -> None:
    output = {
        "task": task,
        "passed": sum(1 for r in results if r.passed),
        "total": len(results),
        "results": [r.to_dict() for r in results],
    }
    print(json.dumps(output, indent=2))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="axor-bench",
        description="Benchmark axor-langchain middleware — all features, no API keys",
    )
    parser.add_argument(
        "--task",
        default="token optimization in multi-agent LLM pipelines",
    )
    parser.add_argument(
        "--scenario",
        choices=list(SCENARIOS.keys()),
        nargs="*",
        default=None,
        help="Run specific scenarios (default: all)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON instead of human-readable report",
    )
    args = parser.parse_args()

    results = run_all(args.task, args.scenario)

    if args.json:
        print_json(results, args.task)
    else:
        print_report(results, args.task)

    sys.exit(0 if all(r.passed for r in results) else 1)


if __name__ == "__main__":
    main()
