"""
axor-langchain live benchmark — real LangGraph pipeline with real API calls.

Runs a 3-node research pipeline (planner -> researcher -> writer)
twice — without and with AxorMiddleware — and compares real tokens
from usage_metadata.

Requirements:
    pip install langchain langchain-anthropic langchain-openai langgraph axor-langchain

Run:
    # Anthropic
    export ANTHROPIC_API_KEY=sk-ant-...
    python benchmark/live_graph.py --provider anthropic

    # OpenAI
    export OPENAI_API_KEY=sk-...
    python benchmark/live_graph.py --provider openai

    # Custom task + multiple runs for statistical reliability
    python benchmark/live_graph.py --provider anthropic \\
        --task "impact of context compression on LLM agent cost" \\
        --mode auto --turns 6 --runs 3

Flags:
    --provider   anthropic | openai  (default: anthropic)
    --model      model ID (default: claude-sonnet-4-6 / gpt-4.1-mini)
    --task       research task
    --mode       auto | minimal | moderate | broad
    --turns      prior history turns (default: 6)
    --runs       number of runs for averaging (default: 1)
    --no-axor    baseline only (skip axor run)
    --axor-only  axor run only (skip baseline)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "axor-core"))


# ── Dependency check ───────────────────────────────────────────────────────────

def _require(pkg: str, install: str | None = None) -> None:
    import importlib
    try:
        importlib.import_module(pkg)
    except ImportError:
        hint = install or pkg
        print(f"\n  ✗ Missing: {pkg}\n  Install: pip install {hint}\n")
        sys.exit(1)

_require("langchain",          "langchain>=1.0.0")
_require("langgraph",          "langgraph>=1.0.0")


# ── Token accounting ───────────────────────────────────────────────────────────

@dataclass
class NodeUsage:
    node:          str
    input_tokens:  int = 0
    output_tokens: int = 0
    latency_ms:    float = 0.0
    tool_calls:    int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class RunResult:
    label:    str                         # "without axor" / "with axor"
    nodes:    list[NodeUsage] = field(default_factory=list)
    error:    str | None = None
    elapsed_s: float = 0.0

    @property
    def total_input(self) -> int:
        return sum(n.input_tokens for n in self.nodes)

    @property
    def total_output(self) -> int:
        return sum(n.output_tokens for n in self.nodes)

    @property
    def total_tokens(self) -> int:
        return self.total_input + self.total_output


# ── Tool definitions ───────────────────────────────────────────────────────────

def _make_tools():
    from langchain.tools import tool

    @tool
    def search_web(query: str) -> str:
        """Search the web for recent information on a topic.
        Returns a list of relevant sources and summaries."""
        # Simulated realistic search output — no real HTTP needed
        return f"""Search results for: "{query}"

[1] Academic paper: "Systematic Review of {query}" (2025)
    Authors: Chen, Wang, Liu  |  Journal: NeurIPS 2025
    Summary: Comprehensive analysis covering 847 studies. Found 34.7%
    efficiency gains with systematic governance across all pipeline stages.
    Key finding: token accumulation follows O(n²) without compression.

[2] Industry report: "{query} — State of the Market 2025"
    Published by: Gartner Research  |  Pages: 142
    Summary: 78% of enterprises face context window overflow in production.
    Hybrid ML+rule systems outperform pure approaches by 2.3x at 60% cost.

[3] Blog post: "How we cut LLM costs 70% in our agent pipeline"
    Company: Stripe Engineering  |  Date: March 2025
    Summary: Implemented context compression. Went from $12K to $3.6K/month.
    Tool output truncation was the single biggest win.

[4] arXiv preprint: "ContextEval: Benchmarking Compression in Agentic Systems"
    Date: April 2025  |  Citations: 89
    Summary: Evaluated 12 compression strategies. Fragment-aware approaches
    preserve 94.3% quality while reducing tokens by 67-82%.

[5] GitHub discussion: langchain-ai/langchain #33441
    Topic: AgentMiddleware tool hooks  |  Status: Merged
    Summary: Added wrap_tool_call hook. Enables per-tool latency tracking."""

    @tool
    def analyze_data(dataset: str, metric: str) -> str:
        """Analyze a dataset and return key statistics for a given metric."""
        return f"""Analysis of '{dataset}' on metric '{metric}':

Statistical summary:
  n = 1,247 observations  |  Period: Jan 2024 – Apr 2025
  Mean:    {42.3:.1f}     |  Median: {38.7:.1f}
  Std dev: {12.4:.1f}     |  IQR:    [{31.2:.1f}, {52.8:.1f}]
  Min:     {8.1:.1f}      |  Max:    {97.6:.1f}

Trend: +18.4% YoY (p < 0.001, ANOVA)
Outliers: 23 detected via IQR method, removed from main analysis.

Segmentation:
  Group A (n=623): mean={48.1:.1f}, significantly higher (p=0.003)
  Group B (n=624): mean={36.5:.1f}
  Effect size: Cohen's d = 0.72 (large effect)

Conclusion: Strong signal in the data supporting the research hypothesis."""

    @tool
    def summarize_findings(findings: str, max_words: int = 200) -> str:
        """Summarize research findings into a concise executive summary."""
        word_count = len(findings.split())
        return f"""Executive Summary ({min(word_count, max_words)} words):

The research presents compelling evidence for systematic approaches to the
studied problem. Three key themes emerge across sources:

1. EFFICIENCY: Systematic methods outperform ad-hoc by 40-60% (strong consensus,
   n=12 independent studies). Effect sizes range from moderate to large.

2. COST: Organizations report 50-70% cost reduction after implementation.
   Tool output compression is the highest-ROI single intervention.

3. QUALITY: Compression at 67-82% token reduction maintains 94%+ output quality
   as measured by human evaluators. Semantic prioritization is key.

Confidence level: HIGH — consistent findings across peer-reviewed and industry sources.
Recommended action: Implement immediately, measure results within 30 days."""

    return [search_web, analyze_data, summarize_findings]


# ── Agent node builders ────────────────────────────────────────────────────────

def _make_planner_agent(model, tools, middleware=None):
    from langchain.agents import create_agent
    kwargs = dict(
        system_prompt=(
            "You are a research planner. Given a topic, call search_web once "
            "to understand the landscape, then write a concise 3-step research plan. "
            "Be brief — 150 words max."
        ),
        tools=tools,
    )
    if middleware:
        kwargs["middleware"] = middleware
    return create_agent(model, **kwargs)


def _make_researcher_agent(model, tools, middleware=None):
    from langchain.agents import create_agent
    kwargs = dict(
        system_prompt=(
            "You are a researcher. Use search_web and analyze_data to gather "
            "evidence. Make exactly 2 tool calls, then write findings. Be concise."
        ),
        tools=tools,
    )
    if middleware:
        kwargs["middleware"] = middleware
    return create_agent(model, **kwargs)


def _make_writer_agent(model, tools, middleware=None):
    from langchain.agents import create_agent
    kwargs = dict(
        system_prompt=(
            "You are a technical writer. Use summarize_findings once, then write "
            "a final report. Max 200 words. Be direct."
        ),
        tools=tools,
    )
    if middleware:
        kwargs["middleware"] = middleware
    return create_agent(model, **kwargs)


# ── Prior history builder ──────────────────────────────────────────────────────

def _make_prior_history(task: str, n_turns: int) -> list:
    """
    Build realistic prior conversation history.
    This is the key factor: agents in prod accumulate history across calls.
    """
    from langchain_core.messages import HumanMessage, AIMessage

    history = []
    subtopics = [
        f"background on {task}",
        f"recent developments in {task}",
        f"key challenges of {task}",
        f"cost implications of {task}",
        f"best practices for {task}",
        f"case studies of {task}",
        f"future outlook for {task}",
        f"technical details of {task}",
    ]
    tool_summaries = [
        "Found 5 relevant papers. Key finding: 34.7% efficiency gain with governance.",
        "Analysis complete: n=1247, mean=42.3, strong YoY trend (+18.4%).",
        "Summarized 8 sources. Three themes: efficiency, cost, quality.",
        "Search returned 4 high-quality industry reports from 2024-2025.",
        "Data shows bimodal distribution. Group A significantly outperforms Group B.",
    ]
    assistant_replies = [
        "Based on my research, the evidence strongly supports systematic approaches. "
        "I found 5 high-quality sources with consistent findings across methodologies. "
        "The primary conclusion is that governance controls yield 40-60% efficiency gains.",
        "The data analysis reveals clear trends. Statistical significance is high (p<0.001). "
        "I recommend focusing on the top 3 interventions based on effect size.",
        "My research plan: (1) literature review, (2) data analysis, (3) synthesis. "
        "I'll prioritize peer-reviewed sources and cross-validate findings.",
    ]

    for i in range(min(n_turns, len(subtopics))):
        history.append(HumanMessage(content=f"Research subtopic: {subtopics[i]}"))
        history.append(AIMessage(
            content=assistant_replies[i % len(assistant_replies)],
            tool_calls=[],
        ))
        # Add a tool result in history
        if i < len(tool_summaries):
            from langchain_core.messages import ToolMessage
            tc_id = f"prior_tc_{i}"
            # insert AI with tool call before the tool result
            history[-1] = AIMessage(
                content="",
                tool_calls=[{"name": "search_web", "args": {"query": subtopics[i]}, "id": tc_id}],
            )
            history.append(ToolMessage(
                content=tool_summaries[i % len(tool_summaries)],
                tool_call_id=tc_id,
            ))
            history.append(AIMessage(
                content=assistant_replies[i % len(assistant_replies)],
                tool_calls=[],
            ))

    return history


# ── Usage extraction ───────────────────────────────────────────────────────────

def _extract_usage(result: dict) -> tuple[int, int]:
    """Pull input/output tokens from agent invocation result."""
    messages = result.get("messages", [])
    total_in = total_out = 0
    for msg in messages:
        usage = getattr(msg, "usage_metadata", None)
        if not usage:
            continue
        # usage_metadata can be a dict or an object with attributes
        if isinstance(usage, dict):
            total_in  += usage.get("input_tokens", 0) or 0
            total_out += usage.get("output_tokens", 0) or 0
        else:
            total_in  += getattr(usage, "input_tokens", 0) or 0
            total_out += getattr(usage, "output_tokens", 0) or 0
    return total_in, total_out


def _count_tool_calls(result: dict) -> int:
    from langchain_core.messages import AIMessage as AI
    return sum(
        len(getattr(m, "tool_calls", []) or [])
        for m in result.get("messages", [])
        if isinstance(m, AI)
    )


# ── Pipeline runner ────────────────────────────────────────────────────────────

def run_live_pipeline(
    task: str,
    model,
    tools: list,
    prior_turns: int,
    axor=None,
) -> RunResult:
    label = "with axor" if axor else "without axor"
    result = RunResult(label=label)
    t_total = time.monotonic()

    # Build middleware list per agent
    mw = [axor] if axor else []

    planner_agent   = _make_planner_agent(model, tools, mw or None)
    researcher_agent = _make_researcher_agent(model, tools, mw or None)
    writer_agent    = _make_writer_agent(model, tools, mw or None)

    # Prior history — same for both runs (deterministic)
    history = _make_prior_history(task, prior_turns)
    accumulated = list(history)

    plan = research = ""

    # ── Planner ───────────────────────────────────────────────────────────────
    print(f"    [{label}] running planner...", end="", flush=True)
    t0 = time.monotonic()
    try:
        r = planner_agent.invoke({
            "messages": accumulated + [{"role": "user", "content": f"Create a research plan for: {task}"}]
        })
        latency = (time.monotonic() - t0) * 1000
        in_tok, out_tok = _extract_usage(r)
        tc = _count_tool_calls(r)
        result.nodes.append(NodeUsage("planner", in_tok, out_tok, latency, tc))
        plan = r["messages"][-1].content if r.get("messages") else ""
        accumulated += r.get("messages", [])
        print(f" ✓ ({in_tok}in/{out_tok}out tok, {latency:.0f}ms)")
    except Exception as e:
        result.error = f"planner: {e}"
        print(f" ✗ {e}")
        result.elapsed_s = time.monotonic() - t_total
        return result

    # ── Researcher ────────────────────────────────────────────────────────────
    print(f"    [{label}] running researcher...", end="", flush=True)
    t0 = time.monotonic()
    try:
        r = researcher_agent.invoke({
            "messages": accumulated + [{"role": "user", "content": f"Research this plan:\n{plan[:300]}"}]
        })
        latency = (time.monotonic() - t0) * 1000
        in_tok, out_tok = _extract_usage(r)
        tc = _count_tool_calls(r)
        result.nodes.append(NodeUsage("researcher", in_tok, out_tok, latency, tc))
        research = r["messages"][-1].content if r.get("messages") else ""
        accumulated += r.get("messages", [])
        print(f" ✓ ({in_tok}in/{out_tok}out tok, {latency:.0f}ms)")
    except Exception as e:
        result.error = f"researcher: {e}"
        print(f" ✗ {e}")
        result.elapsed_s = time.monotonic() - t_total
        return result

    # ── Writer ────────────────────────────────────────────────────────────────
    print(f"    [{label}] running writer...", end="", flush=True)
    t0 = time.monotonic()
    try:
        r = writer_agent.invoke({
            "messages": accumulated + [{"role": "user", "content": f"Write final report:\n{research[:400]}"}]
        })
        latency = (time.monotonic() - t0) * 1000
        in_tok, out_tok = _extract_usage(r)
        tc = _count_tool_calls(r)
        result.nodes.append(NodeUsage("writer", in_tok, out_tok, latency, tc))
        print(f" ✓ ({in_tok}in/{out_tok}out tok, {latency:.0f}ms)")
    except Exception as e:
        result.error = f"writer: {e}"
        print(f" ✗ {e}")

    result.elapsed_s = time.monotonic() - t_total
    return result


# ── Report ─────────────────────────────────────────────────────────────────────

COSTS = {
    # provider: ($/M input, $/M output)
    "anthropic": (3.0, 15.0),
    "openai":    (0.15, 0.60),   # gpt-4.1-mini
}


def _cost(in_tok: int, out_tok: int, provider: str) -> float:
    cin, cout = COSTS.get(provider, (3.0, 15.0))
    return in_tok / 1e6 * cin + out_tok / 1e6 * cout


def _bar(pct: float, width: int = 20) -> str:
    filled = int(min(pct / 100 * width, width))
    return "\033[32m" + "█" * filled + "\033[0m" + "░" * (width - filled)


def print_live_report(
    task:     str,
    raw:      RunResult,
    axor:     RunResult | None,
    provider: str,
    model_id: str,
    prior_turns: int,
    axor_obj=None,
) -> None:
    W = 74
    cin, cout = COSTS.get(provider, (3.0, 15.0))

    print()
    print("  " + "─" * W)
    print(f"  \033[1m  axor-langchain live benchmark\033[0m")
    print(f"     task:     {task}")
    print(f"     model:    {model_id}  ({provider})")
    print(f"     history:  {prior_turns} prior turns")
    print(f"     pricing:  ${cin}/M input, ${cout}/M output tokens")
    print("  " + "─" * W)
    print()

    # ── Per-node table ─────────────────────────────────────────────────────────
    if axor:
        print(f"  \033[2m{'node':<13}{'raw input':>12}{'axor input':>12}"
              f"{'saved':>9}  {'raw out':>9}{'axor out':>10}\033[0m")
        print(f"  \033[2m{'─'*68}\033[0m")

        total_raw_in = total_axor_in = total_raw_out = total_axor_out = 0
        node_names = [n.node for n in raw.nodes]

        for name in node_names:
            r_node = next((n for n in raw.nodes  if n.node == name), None)
            a_node = next((n for n in axor.nodes if n.node == name), None)
            if not r_node:
                continue

            r_in = r_node.input_tokens
            a_in = a_node.input_tokens if a_node else 0
            r_out = r_node.output_tokens
            a_out = a_node.output_tokens if a_node else 0

            total_raw_in   += r_in
            total_axor_in  += a_in
            total_raw_out  += r_out
            total_axor_out += a_out

            pct = (r_in - a_in) / r_in * 100 if r_in > 0 else 0
            color = "\033[32m" if pct >= 30 else "\033[33m" if pct >= 10 else ""
            reset = "\033[0m"
            print(
                f"  {name:<13}"
                f"{r_in:>10,} tok"
                f"{a_in:>10,} tok"
                f"  {color}{pct:>+.1f}%{reset}"
                f"  {r_out:>7,}"
                f"{a_out:>9,}"
            )

        print(f"  \033[2m{'─'*68}\033[0m")
        total_pct = (total_raw_in - total_axor_in) / total_raw_in * 100 if total_raw_in else 0
        print(
            f"  \033[1m{'TOTAL':<13}\033[0m"
            f"{total_raw_in:>10,} tok"
            f"{total_axor_in:>10,} tok"
            f"  \033[32m{total_pct:>+.1f}%\033[0m"
            f"  {total_raw_out:>7,}"
            f"{total_axor_out:>9,}"
        )
        print()

        # ── Cost ───────────────────────────────────────────────────────────────
        raw_cost  = _cost(total_raw_in,  total_raw_out,  provider)
        axor_cost = _cost(total_axor_in, total_axor_out, provider)
        saved     = raw_cost - axor_cost

        print(f"  \033[1mCost per pipeline run\033[0m")
        print(f"  without axor:  ${raw_cost:.5f}")
        print(f"  with axor:     ${axor_cost:.5f}")
        print(f"  savings:       \033[32m${saved:.5f}\033[0m  ({total_pct:.1f}% input reduction)")
        print()

        print(f"  {'runs/month':<12}{'without':>12}{'with':>12}{'saved':>12}")
        print(f"  \033[2m{'─'*50}\033[0m")
        for scale in [100, 1_000, 10_000]:
            r = raw_cost * scale;  a = axor_cost * scale;  s = r - a
            print(f"  {scale:>10,}     ${r:>9.2f}   ${a:>9.2f}   \033[32m${s:>9.2f}\033[0m")
        print()

        # ── Latency ───────────────────────────────────────────────────────────
        print(f"  \033[1mLatency\033[0m")
        print(f"  without axor:  {raw.elapsed_s:.1f}s total")
        print(f"  with axor:     {axor.elapsed_s:.1f}s total")
        overhead = axor.elapsed_s - raw.elapsed_s
        print(f"  axor overhead: {overhead:+.1f}s  (compression is CPU-only, no extra API calls)")
        print()

        if axor_obj:
            print(f"  \033[1maxor middleware stats\033[0m")
            print(f"  turns processed:  {axor_obj.turns}")
            print(f"  tokens tracked:   {axor_obj.total_tokens_spent:,}")
            engine = "ContextCompressor" if axor_obj._engines_ready else "built-in fallback"
            print(f"  engine:           {engine}")

            # tool stats
            ts = axor_obj.tool_stats
            if ts and ts.total_calls > 0:
                print(f"\n  \033[1mtool stats\033[0m")
                print(f"  {ts.summary()}")
            elif ts:
                print(f"  tool calls:       0 (middleware wrap_tool_call may not have fired)")

            # verification
            if axor_obj.turns == 0:
                print(f"\n  \033[31m  WARNING: axor.turns == 0 — middleware did not process any turns.\033[0m")
                print(f"  \033[31m  This means before_model() hook was not called by the agent.\033[0m")
            print()

    else:
        # Baseline only
        print(f"  \033[1mBaseline (without axor)\033[0m")
        for n in raw.nodes:
            print(f"  {n.node:<13} {n.input_tokens:>8,} in  {n.output_tokens:>6,} out  {n.latency_ms:.0f}ms  {n.tool_calls} tool calls")
        raw_cost = _cost(raw.total_input, raw.total_output, provider)
        print(f"\n  Total: {raw.total_tokens:,} tokens  ${raw_cost:.5f}/run")
        print(f"  Elapsed: {raw.elapsed_s:.1f}s")

    if raw.error:
        print(f"  \033[31m✗ raw run error: {raw.error}\033[0m")
    if axor and axor.error:
        print(f"  \033[31m✗ axor run error: {axor.error}\033[0m")

    print("  " + "─" * W)
    print()


# ── Main ───────────────────────────────────────────────────────────────────────

def _build_model(provider: str, model_id: str | None):
    if provider == "anthropic":
        _require("langchain_anthropic", "langchain-anthropic")
        from langchain_anthropic import ChatAnthropic
        mid = model_id or "claude-sonnet-4-6"
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            print("\n  ✗ ANTHROPIC_API_KEY not set\n")
            sys.exit(1)
        return ChatAnthropic(model=mid, api_key=key, max_tokens=1024), mid

    elif provider == "openai":
        _require("langchain_openai", "langchain-openai")
        from langchain_openai import ChatOpenAI
        mid = model_id or "gpt-4.1-mini"
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            print("\n  ✗ OPENAI_API_KEY not set\n")
            sys.exit(1)
        return ChatOpenAI(model=mid, api_key=key, max_tokens=1024), mid

    else:
        print(f"\n  ✗ Unknown provider: {provider!r}. Use: anthropic | openai\n")
        sys.exit(1)


def main():
    p = argparse.ArgumentParser(
        prog="axor-live-bench",
        description="Live benchmark: real API calls, real token counts",
    )
    p.add_argument("--provider", choices=["anthropic", "openai"], default="anthropic")
    p.add_argument("--model",    default=None, help="Model ID (default: claude-sonnet-4-6 / gpt-4.1-mini)")
    p.add_argument("--task",     default="token optimization in multi-agent LLM pipelines")
    p.add_argument("--mode",     choices=["auto", "minimal", "moderate", "broad"], default="auto")
    p.add_argument("--turns",    type=int, default=6, help="Prior history turns (default: 6)")
    p.add_argument("--runs",     type=int, default=1, help="Number of runs for averaging (default: 1)")
    p.add_argument("--no-axor",  action="store_true", help="Baseline only (no axor run)")
    p.add_argument("--axor-only", action="store_true", help="axor run only (skip baseline)")
    args = p.parse_args()

    model, model_id = _build_model(args.provider, args.model)
    tools = _make_tools()

    from axor_langchain.middleware import AxorMiddleware
    axor_obj = AxorMiddleware(
        compression_mode=args.mode,
        soft_token_limit=200_000,
        allowed_tools=None,
        denied_tools=None,
        # personality=None — agents already have system_prompt via create_agent,
        # setting personality here would create duplicate system messages
        tool_max_retries=1,
        tool_retry_delay=0.5,
        tool_error_handler=lambda name, exc: f"[Tool {name} failed: {exc}]",
        track_tool_stats=True,
        verbose=False,
    )

    n_runs = max(1, args.runs)
    print()
    print(f"  axor-langchain live benchmark")
    print(f"  provider={args.provider}  model={model_id}  turns={args.turns}  mode={args.mode}  runs={n_runs}")
    print()

    raw_results: list[RunResult] = []
    axor_results: list[RunResult] = []

    for run_i in range(n_runs):
        if n_runs > 1:
            print(f"  ══ Run {run_i + 1}/{n_runs} ════════════════════════════════════════")

        if not args.axor_only:
            print(f"  ── without axor ─────────────────────────────────────────")
            raw_results.append(run_live_pipeline(
                task=args.task, model=model, tools=tools, prior_turns=args.turns,
            ))

        if not args.no_axor:
            # fresh middleware per run for clean stats
            axor_run = AxorMiddleware(
                compression_mode=args.mode,
                soft_token_limit=200_000,
                allowed_tools=None,
                denied_tools=None,
                tool_max_retries=1,
                tool_retry_delay=0.5,
                tool_error_handler=lambda name, exc: f"[Tool {name} failed: {exc}]",
                track_tool_stats=True,
                verbose=False,
            )
            print(f"  ── with axor ────────────────────────────────────────────")
            axor_results.append(run_live_pipeline(
                task=args.task, model=model, tools=tools, prior_turns=args.turns, axor=axor_run,
            ))
            # keep last axor instance for stats
            axor_obj = axor_run

    # Use last run for detailed report, show averages if multiple runs
    baseline = raw_results[-1] if raw_results else (axor_results[-1] if axor_results else None)
    comparison = axor_results[-1] if raw_results and axor_results else None

    print_live_report(
        task=args.task,
        raw=baseline,
        axor=comparison,
        provider=args.provider,
        model_id=model_id,
        prior_turns=args.turns,
        axor_obj=axor_obj if axor_results else None,
    )

    # Print averages for multi-run
    if n_runs > 1:
        print(f"  \033[1mMulti-run summary ({n_runs} runs)\033[0m")
        if raw_results:
            avg_in = sum(r.total_input for r in raw_results) / n_runs
            avg_out = sum(r.total_output for r in raw_results) / n_runs
            print(f"  without axor — avg input: {avg_in:,.0f} tok, avg output: {avg_out:,.0f} tok")
        if axor_results:
            avg_in = sum(r.total_input for r in axor_results) / n_runs
            avg_out = sum(r.total_output for r in axor_results) / n_runs
            print(f"  with axor    — avg input: {avg_in:,.0f} tok, avg output: {avg_out:,.0f} tok")
        if raw_results and axor_results:
            raw_avg = sum(r.total_input for r in raw_results) / n_runs
            axor_avg = sum(r.total_input for r in axor_results) / n_runs
            if raw_avg > 0:
                pct = (raw_avg - axor_avg) / raw_avg * 100
                print(f"  avg input reduction: \033[32m{pct:.1f}%\033[0m")
        print()


if __name__ == "__main__":
    main()
