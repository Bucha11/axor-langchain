# axor-langchain

[![CI](https://github.com/Bucha11/axor-langchain/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Bucha11/axor-langchain/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/axor-langchain?cacheSeconds=300)](https://pypi.org/project/axor-langchain/)
[![Python](https://img.shields.io/pypi/pyversions/axor-langchain?cacheSeconds=300)](https://pypi.org/project/axor-langchain/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Production middleware for LangChain agents that cuts context growth without
rewriting your graph.

Axor compresses stale tool-heavy history, keeps fresh tool outputs intact,
enforces token budgets, filters tools, and optionally caches deterministic tool
results. Add it to `create_agent()` as middleware and keep your existing tools,
models, prompts, and LangGraph topology.

Validated on live hard-agent benchmarks:

- OpenAI aggressive: **77.0% aggregate cost savings**, **0.91 average judge score** (3-run average)
- OpenAI cautious: **69.9% aggregate cost savings**, **0.92 average judge score** (3-run average)
- Anthropic aggressive: **35.3% aggregate cost savings**, **0.94 average judge score** (3-run average)
- Anthropic cautious: **30.0% aggregate cost savings**, **0.96 average judge score** (3-run average)

## Why Axor

Long-running agents do not usually fail because one prompt is large. They fail
because every turn carries yesterday's logs, traces, search results, and
intermediate reasoning into tomorrow's model call.

Axor focuses on that production path:

- reduce repeated input tokens in multi-tool and multi-step agents
- preserve the newest tool results so agents do not re-query unnecessarily
- make cost controls explicit with soft and hard token limits
- restrict dangerous or irrelevant tools per agent
- cache read-only tool outputs across LangGraph invocations
- validate savings with a live benchmark and LLM-as-judge quality check

## Install

```bash
pip install axor-langchain
```

Provider packages are optional:

```bash
pip install "axor-langchain[anthropic]"
pip install "axor-langchain[openai]"
pip install "axor-langchain[providers]"
```

## Quick Start

```python
from langchain.agents import create_agent
from axor_langchain import AxorMiddleware

axor = AxorMiddleware(
    optimization_profile="cautious",
    soft_token_limit=80_000,
    hard_token_limit=120_000,
)

agent = create_agent(
    "anthropic:claude-sonnet-4-5",
    tools=tools,
    middleware=[axor],
)

result = await agent.ainvoke({
    "messages": [("user", "Investigate the checkout latency incident.")]
})

print(f"tokens spent: {axor.total_tokens_spent}")
```

## Core Features

| Feature | Production Behavior |
|---------|---------------------|
| Context compression | Uses axor-core `ContextCompressor` with pinned, knowledge, working, and ephemeral fragment semantics |
| Fresh-tool protection | Keeps the most recent tool outputs verbatim to avoid retry loops after compression |
| Policy selection | Uses axor-core task classification unless an explicit compression mode is set |
| Tool governance | Applies allowlists and denylists before tools reach the model |
| Budget guardrails | Estimates input before the call, then records provider-reported usage after the call |
| Tool cache | Opt-in cache for deterministic read-only tools, persisted in LangGraph state |
| Telemetry | Off by default; local or remote anonymized telemetry is explicit opt-in |

## Optimization Profiles

Use profiles first; override individual knobs only after measuring.

| Profile | Use When | Settings | Expected Tradeoff |
|---------|----------|----------|-------------------|
| `cautious` | Initial rollout, regulated workflows, quality-sensitive agents | policy-selected compression, last 2 tool results kept verbatim, all tools available to the model | lower savings, wider quality margin |
| `aggressive` | High-volume hard agents with large tool outputs | aggressive compression, last 1 tool result kept verbatim, top-K=8 task-relevant tools, deduplicates repeated tool calls in old turns | highest measured savings, requires quality validation |

```python
quality_first = AxorMiddleware(optimization_profile="cautious")
cost_first = AxorMiddleware(optimization_profile="aggressive")

custom = AxorMiddleware(
    optimization_profile="aggressive",
    recent_tools_window=2,
    compression_mode="balanced",
)
```

Explicit `recent_tools_window` and `compression_mode` values override the
profile defaults.

## Tool Governance

Run different agents with different tool surfaces:

```python
research_axor = AxorMiddleware(
    allowed_tools=["search", "read_file", "lookup_doc"],
)

review_axor = AxorMiddleware(
    denied_tools=["bash", "write_file", "delete_file"],
)
```

## Budget Controls

```python
axor = AxorMiddleware(
    soft_token_limit=80_000,
    hard_token_limit=120_000,
    verbose=True,
)
```

The hard limit is a pre-call gate. Axor estimates the next input size and raises
`BudgetExceededError` before sending an over-budget request. Actual accounting
uses the provider's `usage_metadata` after each model call.

## Tool Result Cache

Tool caching is intentionally opt-in. Use it only for deterministic read-only
tools:

```python
axor = AxorMiddleware(
    cache_tools=["read_file", "lookup_doc"],
    max_tool_cache_entries=100,
)
```

Cache entries are keyed by tool name and arguments. The cache lives in
`AxorState.tool_result_cache`, so LangGraph checkpointing can preserve it across
invocations under the same `thread_id`.

## Anthropic Prompt Caching

Axor does not write provider-specific `cache_control` markers. Compose it with
LangChain's Anthropic middleware when you want prompt caching:

```python
from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware

agent = create_agent(
    "anthropic:claude-sonnet-4-5",
    tools=tools,
    middleware=[
        AxorMiddleware(optimization_profile="cautious"),
        AnthropicPromptCachingMiddleware(),
    ],
)
```

Order matters: list `AxorMiddleware` **before** `AnthropicPromptCachingMiddleware`
so compression runs first and the Anthropic middleware places `cache_control`
markers on the final, compressed message set. The reverse order can stamp
markers onto messages that Axor then rewrites, dropping the cache hit.

## Telemetry

Telemetry is off by default.

```python
local = AxorMiddleware(telemetry="local")
remote = AxorMiddleware(telemetry="remote")
```

Remote telemetry sends anonymized policy and token metadata only. It does not
send raw prompts, tool arguments, file contents, secrets, user IDs, or session
IDs.

## Configuration

```python
AxorMiddleware(
    soft_token_limit=None,
    hard_token_limit=None,
    allowed_tools=None,
    denied_tools=None,
    personality=None,
    memory_provider=None,
    memory_namespace="axor",
    tool_error_handler=None,
    tool_max_retries=0,
    tool_retry_delay=0.0,
    track_tool_stats=False,
    cache_tools=None,
    max_tool_cache_entries=100,
    optimization_profile=None,  # None | "cautious" | "aggressive"
    token_cost_rates=None,      # optional axor_core.budget.TokenCostRates
    recent_tools_window=None,
    compression_mode=None,      # None/"auto" | "aggressive" | "balanced" | "light"
    tool_selection=None,        # None | "relevance"
    tool_top_k=None,            # cap on tools shown to the model when relevance is on
    tool_min_keep=3,            # floor so relevance never strips below this
    tool_sticky_lookback=4,     # AI turns whose tools are anchored even on low score
    tool_dedup_old_results=None,# replace old duplicate tool results with a pointer
    tool_selection_stable=True, # cache the relevance selection within a user turn
    verbose=False,
    telemetry=None,             # None | "off" | "local" | "remote"
)
```

The `tool_selection`, `tool_top_k`, and `tool_dedup_old_results` knobs are
turned on automatically by `optimization_profile="aggressive"`. Set them
explicitly only after you have a reason to override the profile.

## Benchmark

The supported benchmark is a live hard-agent suite:

```bash
cd axor-langchain
export ANTHROPIC_API_KEY=sk-ant-...

python benchmark/live_hard_agent.py \
  --provider anthropic \
  --task all \
  --prior-turns 10 \
  --tool-kb 10 \
  --axor-profile aggressive \
  --judge \
  --json
```

Run the cautious rollout profile with the same harness:

```bash
python benchmark/live_hard_agent.py \
  --provider anthropic \
  --task all \
  --prior-turns 10 \
  --tool-kb 10 \
  --axor-profile cautious \
  --judge \
  --json
```

It runs each task twice:

- baseline: LangChain `create_agent()` without Axor
- governed: the same agent with `AxorMiddleware`

The benchmark uses real provider calls, realistic prior history, large
deterministic tool outputs, and optional LLM-as-judge quality scoring.

### Tasks

| Task | Measures |
|------|----------|
| `incident_rca` | incident timeline, root cause, blast radius, mitigations |
| `security_migration` | OAuth migration planning, vulnerable paths, rollout, backout |
| `cost_optimization` | model-spend diagnosis for tool-heavy agent workflows |

### Validated Results

Anthropic aggressive profile, `task=all`, `--judge` (model=`claude-sonnet-4-6`,
`prior-turns=10`, `tool-kb=10`; auto-fit narrowed `prior-kb` to 2 and `tool-kb`
to 3 to fit Anthropic input TPM). Averaged over 3 independent runs:

| Task | Judge Score | Verdict | Input Savings | Total Savings | Cost Savings |
|------|-------------|---------|---------------|---------------|--------------|
| `incident_rca` | 0.96 | equivalent | 55.4% | 51.9% | 42.0% |
| `security_migration` | 0.92 | equivalent | 34.4% | 32.8% | 29.0% |
| `cost_optimization` | 0.94 | equivalent | 45.1% | 42.0% | 32.7% |
| **Aggregate** | **0.94 avg** | equivalent | **47.2%** | **44.0%** | **35.3%** |

Per-task numbers carry ±10pp run-to-run variance on this profile because the
baseline tool-call count and depth are non-deterministic; the aggregate is
stable across runs. `security_migration` consistently shows lower compression
opportunity because the baseline tends to converge on a tighter scope here.

Anthropic cautious profile, `task=all`, `--judge` (model=`claude-sonnet-4-6`,
`prior-turns=10`, `tool-kb=10`; auto-fit narrowed `prior-kb` to 2 and `tool-kb`
to 3 to fit Anthropic input TPM). Averaged over 3 independent runs:

| Task | Judge Score | Verdict | Input Savings | Total Savings | Cost Savings |
|------|-------------|---------|---------------|---------------|--------------|
| `incident_rca` | 0.98 | equivalent | 45.0% | 41.9% | 32.8% |
| `security_migration` | 0.95 | equivalent | 30.4% | 28.4% | 22.6% |
| `cost_optimization` | 0.95 | equivalent | 44.1% | 41.3% | 33.1% |
| **Aggregate** | **0.96 avg** | equivalent | **41.9%** | **38.8%** | **30.0%** |

`security_migration` is the lowest-savings task on this profile because its
baseline tends to converge on a tight scope; on a 27K-token baseline the
governance overhead can briefly exceed the compression gain. On larger
payloads (`incident_rca`, `cost_optimization`) the profile holds ~32–33%
cost reduction.

OpenAI aggressive profile, `task=all`, `--judge` (model=`gpt-4.1-mini`,
`prior-turns=10`, `tool-kb=10`). Averaged over 3 independent runs:

| Task | Judge Score | Verdict | Input Savings | Total Savings | Cost Savings |
|------|-------------|---------|---------------|---------------|--------------|
| `incident_rca` | 0.92 | equivalent | 80.9% | 79.6% | 76.1% |
| `security_migration` | 0.92 | equivalent | 81.1% | 80.2% | 77.6% |
| `cost_optimization` | 0.88 | mixed | 80.7% | 79.8% | 77.2% |
| **Aggregate** | **0.91 avg** | mostly equivalent | **80.9%** | **79.9%** | **77.0%** |

`cost_optimization` lands on `minor_drift` in 2 of 3 runs under aggressive
compression — the governed response trims concrete actions (rollback steps,
deadline propagation, circuit-breaker callouts) while preserving the
diagnosis. If you need the action list intact for this task class, use
`cautious` instead.

OpenAI cautious profile, `task=all`, `--judge` (model=`gpt-4.1-mini`,
`prior-turns=10`, `tool-kb=10`). Averaged over 3 independent runs:

| Task | Judge Score | Verdict | Input Savings | Total Savings | Cost Savings |
|------|-------------|---------|---------------|---------------|--------------|
| `incident_rca` | 0.93 | equivalent | 73.3% | 72.2% | 69.2% |
| `security_migration` | 0.92 | equivalent | 73.9% | 73.1% | 70.6% |
| `cost_optimization` | 0.92 | equivalent | 72.6% | 71.9% | 70.0% |
| **Aggregate** | **0.92 avg** | equivalent | **73.3%** | **72.4%** | **69.9%** |

OpenAI showed the strongest aggregate cost savings on this benchmark.
Under the aggressive profile, `cost_optimization` lands on `minor_drift`
in 2 of 3 runs — a real cost-vs-quality tradeoff for that task class, not
run-to-run noise. The Anthropic cautious profile produced the highest
average judge score (0.96) with all tasks `equivalent`, at the cost of a
smaller percentage cost reduction.

Treat profiles as deployment presets, not universal quality rankings; validate
with `--judge` on your own workload.

Profile decision guide:

| Profile | Primary Goal | Use It For | Publishable Measured Result |
|---------|--------------|------------|-----------------------------|
| `cautious` | preserve quality first | staging rollout, first production cohort, sensitive agents | OpenAI: 70.9% cost savings; Anthropic: 30.6%; all tasks equivalent |
| `aggressive` | maximize savings with judge guardrails | high-volume production agents after validation | OpenAI: 77.2% cost savings; Anthropic: 48.5%; all tasks equivalent |

Recommended rollout path:

1. Start with `optimization_profile="cautious"` in staging.
2. Run the benchmark with `--judge` on representative tasks.
3. Move high-volume agents to `optimization_profile="aggressive"` when quality
   scores stay acceptable.

## Requirements

- Python 3.11+
- `langchain >= 1.0.0`
- `langgraph >= 1.0.0`
- `axor-core`

## License

MIT
