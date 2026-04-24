# axor-langchain

[![CI](https://github.com/Bucha11/axor-langchain/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Bucha11/axor-langchain/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/axor-langchain?cacheSeconds=300)](https://pypi.org/project/axor-langchain/)
[![Python](https://img.shields.io/pypi/pyversions/axor-langchain?cacheSeconds=300)](https://pypi.org/project/axor-langchain/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Cut token costs 40–80% in LangChain multi-agent pipelines.**

One middleware. No graph changes. Works with any `create_agent()` agent.

![demo](docs/demo.svg)

---

## The problem

LangChain agents accumulate messages. By turn 10 you're paying for:
- Tool outputs from 8 turns ago that nobody needs
- Repeated context that hasn't changed
- Intermediate reasoning that's already been acted on

A 10-node research pipeline can balloon from 5k to 80k tokens by the last node — and you're billed for all of it on every API call.

---

## Installation

```bash
pip install axor-langchain
```

---

## Quick start

```python
from langchain.agents import create_agent
from axor_langchain import AxorMiddleware

# before: bare agent
agent = create_agent("anthropic:claude-sonnet-4-5", tools=tools)

# after: governed agent — one line change
axor = AxorMiddleware(soft_token_limit=100_000, verbose=True)
agent = create_agent(
    "anthropic:claude-sonnet-4-5",
    tools=tools,
    middleware=[axor],
)

result = await agent.ainvoke({"messages": [("user", "research transformers")]})
print(f"Tokens spent: {axor.total_tokens_spent}")
```

---

## What it does

### Context compression

Before each model call, `AxorMiddleware` compresses the message history based on session length:

| Session length | Mode | Window | Tool output cap |
|---------------|------|--------|----------------|
| ≤ 6 messages | broad | all | 8,000 chars |
| 7–20 messages | moderate | last 16 | 2,000 chars |
| 21+ messages | minimal | last 6 | 800 chars |

The longer the session, the more aggressively old context is compressed. Recent messages are always kept. System messages are never dropped.

**Typical savings:**

```
Turn  1:  1,200 tokens  (no compression yet)
Turn  5:  1,800 tokens  (moderate: old tools truncated)
Turn 10:  2,100 tokens  (minimal: only recent 6 messages)
Turn 20:  2,300 tokens  (stable — doesn't keep growing)

Without axor:
Turn 20: 45,000 tokens  (full history accumulated)
```

### Tool governance

Filter which tools each agent can call — without changing the graph:

```python
# research agent: read + search only, no write/bash
axor = AxorMiddleware(
    allowed_tools=["search", "read", "web_search"],
)

# audit agent: read only
axor = AxorMiddleware(
    denied_tools=["write", "bash", "delete"],
)
```

### Budget tracking

Hard stop when token limit is reached — no surprise bills:

```python
axor = AxorMiddleware(
    soft_token_limit=80_000,   # log warning
    hard_token_limit=100_000,  # stop agent, return partial result
)
```

### Pinned personality

Personality is always the first system message — survives compression:

```python
axor = AxorMiddleware(
    personality="You are a security-focused code reviewer. "
                "Always check for injection risks and hardcoded secrets.",
)
```

### Cross-session memory (optional)

```bash
pip install axor-langchain[memory]
```

```python
from axor_memory_sqlite import SQLiteMemoryProvider

provider = SQLiteMemoryProvider("~/.axor/memory.db")
axor = AxorMiddleware(
    memory_provider=provider,
    memory_namespace="research-agent",
)
# after each session: last assistant message saved to SQLite
# next session: load with provider.load(MemoryQuery(...))
```

### Anonymous telemetry (opt-in)

```bash
pip install axor-langchain[telemetry]
```

```python
# explicit kwarg
axor = AxorMiddleware(telemetry="local")    # append to local JSONL queue
axor = AxorMiddleware(telemetry="remote")   # also ship to telemetry.useaxor.net

# or env (no code change)
# AXOR_TELEMETRY=local  or  AXOR_TELEMETRY=remote
```

**What gets sent** (only with `remote`, only when opted in):

- chosen signal (`focused_generative`, etc), classifier confidence, tokens spent
- 128-int MinHash fingerprint of the task — non-reversible
- whether policy was corrected mid-run, `axor_version`

**Never sent:** raw task text, file contents, tool arguments, secrets, user/session IDs. IP is hashed SHA-256 truncated to 16 chars, used only for rate-limit buckets.

Live community aggregates and the full data contract:
[telemetry.useaxor.net/stats](https://telemetry.useaxor.net/stats). Suppress
the one-time opt-in notice with `AXOR_NO_BANNER=1`.

Default is **off** — nothing leaves your machine without an explicit opt-in.

### Small context bypass

By default, contexts under 4,000 tokens skip the compression pipeline entirely. This avoids overhead on small/early turns where compression can't save more than it costs:

```python
# default: auto-bypass for small contexts (recommended)
axor = AxorMiddleware(soft_token_limit=100_000)

# disable bypass — always compress (aggressive savings, may add overhead on small turns)
axor = AxorMiddleware(bypass_token_threshold=0)

# custom threshold
axor = AxorMiddleware(bypass_token_threshold=8000)
```

Budget tracking and tool governance still apply even when compression is bypassed.

**Impact on savings (real data from claude-sonnet-4-6 benchmark):**

| | Without bypass | With bypass (4000) |
|--|---------------|-------------------|
| Total savings (4t + 8t combined) | +26.4% | **+24.8%** |
| Risk of negative savings on small contexts | Yes (-9% at 6t) | **No** (0% — passed through) |
| Large context savings (8t+) | +26-48% | **+26-48%** (unchanged) |

Bypass trades ~1.6% total savings for stable, predictable behavior — you never pay more than without axor.

---

## LangGraph integration

Works with any LangGraph `StateGraph` that uses LangChain agents as nodes:

```python
from langgraph.graph import StateGraph, END
from langchain.agents import create_agent
from axor_langchain import AxorMiddleware

# each node gets its own governance config
research_axor = AxorMiddleware(
    allowed_tools=["search", "web_search"],
    soft_token_limit=50_000,
    verbose=True,
)
writer_axor = AxorMiddleware(
    allowed_tools=["read", "write"],
    soft_token_limit=30_000,
)

research_agent = create_agent(
    "anthropic:claude-sonnet-4-5",
    tools=[search_tool, web_search_tool],
    middleware=[research_axor],
)
writer_agent = create_agent(
    "anthropic:claude-sonnet-4-5",
    tools=[read_tool, write_tool],
    middleware=[writer_axor],
)

workflow = StateGraph(State)
workflow.add_node("research", research_agent)
workflow.add_node("write",    writer_agent)
workflow.add_edge("research", "write")
workflow.add_edge("write", END)

app = workflow.compile()
result = await app.ainvoke({"messages": [...]})

print(f"Research tokens: {research_axor.total_tokens_spent}")
print(f"Writer tokens:   {writer_axor.total_tokens_spent}")
```

Per-node governance: each agent compresses its own context independently.

---

## Configuration reference

```python
AxorMiddleware(
    soft_token_limit=None,           # int | None — warning threshold
    hard_token_limit=None,           # int | None — stop threshold (default: soft * 1.5)
    compression_mode="auto",         # "auto" | "minimal" | "moderate" | "broad"
    bypass_token_threshold=4000,     # int — skip compression below this token count
    allowed_tools=None,              # list[str] | None — whitelist
    denied_tools=None,               # list[str] | None — blacklist
    personality=None,                # str | None — pinned system message
    memory_provider=None,            # MemoryProvider | None
    memory_namespace="axor",         # str
    tool_error_handler=None,         # Callable[[str, Exception], str] | None
    tool_max_retries=0,              # int — extra retry attempts
    tool_retry_delay=0.0,            # float — seconds between retries
    track_tool_stats=False,          # bool — per-tool call/latency/error tracking
    verbose=False,                   # bool — log governance decisions
    telemetry=None,                  # "off" | "local" | "remote" | None (AXOR_TELEMETRY env)
)
```

---

## Difference from axor-claude

| | axor-claude | axor-langchain |
|--|-------------|----------------|
| Provider | Anthropic only | any (OpenAI, Anthropic, Google…) |
| Framework | axor-core GovernedSession | LangChain create_agent() |
| Governance depth | full (context shaping, IntentLoop) | middleware (message compression, tool filter) |
| Best for | standalone coding agents | multi-agent LangGraph pipelines |

---

## Requirements

- Python 3.11+
- `langchain >= 1.0.0`
- `langgraph >= 1.0.0`

---

## License

MIT

---

## Benchmarks

### Live results (claude-sonnet-4-6, 3-node research pipeline)

Real API calls, real `usage_metadata` token counts. Pipeline: planner → researcher → writer.
Default `bypass_token_threshold=4000` — small contexts pass through without compression.

**Per-node breakdown (8 turns, auto mode):**

| Node | Without axor | With axor | Saved |
|------|-------------|-----------|-------|
| planner | 13,678 tok | 7,112 tok | **48.0%** |
| researcher | 27,677 tok | 19,750 tok | **28.6%** |
| writer | 44,963 tok | 36,811 tok | **18.1%** |
| **TOTAL** | **86,318 tok** | **63,673 tok** | **26.2%** |

Writer sees all accumulated context from planner + researcher — this is where token explosion happens in production.

**Across configurations:**

| Prior turns | Mode | Without axor | With axor | Savings | $/10K runs saved |
|------------|------|-------------|-----------|---------|-----------------|
| 4 turns | auto | 28,366 tok | 20,717 tok | **27.0%** | **$274** |
| 8 turns | auto | 86,318 tok | 63,673 tok | **26.2%** | **$733** |
| 8 turns | minimal | 65,243 tok | 52,451 tok | **19.6%** | **$438** |

> Pricing: claude-sonnet-4-6 @ $3/M input, $15/M output tokens.
> Results vary between runs due to LLM non-determinism. Use `--runs 3` for averaged results.

**Bypass impact (calculated from real data across 4t + 8t runs):**

| | Without bypass | With bypass (default) |
|--|---------------|----------------------|
| Total savings | +26.4% | **+24.8%** |
| Negative savings risk | Yes | **No** |
| Large context savings | +26-48% | +26-48% (same) |

~1.6% less total savings, but guaranteed no overhead on small contexts.

### Simulated benchmark (no API key needed)

Tests all middleware features: compression, tool governance, budget, tool retry, bypass detection.

```bash
python benchmark/run.py                         # all 17 scenarios
python benchmark/run.py --scenario bypass       # test bypass only
python benchmark/run.py --json                  # CI-friendly output
```

### Live benchmark

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python benchmark/live_graph.py --provider anthropic --turns 8
python benchmark/live_graph.py --provider anthropic --runs 3  # averaged

# OpenAI
export OPENAI_API_KEY=sk-...
python benchmark/live_graph.py --provider openai
```

| Flag | Default | Description |
|------|---------|-------------|
| `--provider` | `anthropic` | `anthropic` or `openai` |
| `--model` | `claude-sonnet-4-6` / `gpt-4.1-mini` | Override model |
| `--task` | research topic | Task for the agent |
| `--mode` | `auto` | Compression mode |
| `--turns` | `6` | Prior history turns |
| `--runs` | `1` | Number of runs for averaging |
| `--no-axor` | — | Baseline only |
| `--axor-only` | — | axor run only |
