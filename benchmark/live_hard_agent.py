#!/usr/bin/env python3
"""
Live hard-agent benchmark for axor-langchain.

This benchmark uses a real provider model, a production-like LangChain agent,
many tools, large deterministic tool outputs, and a large valid prior history.
It runs the same hard task twice:

  1. baseline: create_agent(...), no AxorMiddleware
  2. governed: create_agent(..., middleware=[AxorMiddleware(...)])

The tools are synthetic but realistic: they do not call the network, yet they
return large log excerpts, traces, repository files, metrics, dependency
audits, runbooks, incident docs, and security findings. That keeps the run
reproducible while still exercising the expensive part of real agents: model
calls over large accumulated context and tool outputs.

Examples:
    export ANTHROPIC_API_KEY=sk-ant-...
    python benchmark/live_hard_agent.py --provider anthropic --task incident_rca

    export OPENAI_API_KEY=sk-...
    python benchmark/live_hard_agent.py --provider openai --task security_migration

    python benchmark/live_hard_agent.py --provider anthropic --task all \
        --prior-turns 10 --tool-kb 10 --json

By default the script auto-fits synthetic payload sizes to a live API budget
so it can run on common Anthropic 30k input-token/minute limits. Disable that
with --no-auto-fit when running under a higher tier.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


COSTS_PER_M = {
    "anthropic": (3.00, 15.00),
    "openai": (0.15, 0.60),
}


TASKS = {
    "incident_rca": (
        "A checkout latency incident caused p95 to spike from 420ms to 4.8s "
        "after a release. Produce a production-grade RCA with timeline, root "
        "cause, contributing factors, blast radius, immediate mitigation, and "
        "a prioritized prevention plan. Use concrete evidence from tools."
    ),
    "security_migration": (
        "Plan a high-risk migration from API keys to short-lived OAuth tokens "
        "for a multi-tenant billing platform. Identify vulnerable code paths, "
        "dependency risks, rollout phases, backout plan, test strategy, and "
        "operational dashboards. Use concrete evidence from tools."
    ),
    "cost_optimization": (
        "The agent platform's monthly model bill doubled after adding new "
        "research workflows. Diagnose the drivers and propose a production "
        "optimization plan that preserves quality. Use traces, metrics, repo "
        "files, and runbooks as evidence."
    ),
}

BENCHMARK_AXOR_PROFILES = {
    "cautious": {
        "recent_tools_window": 2,
        "compression_mode": "auto",
        "max_tokens": 5000,
    },
    "aggressive": {
        "recent_tools_window": 1,
        "compression_mode": "aggressive",
        "max_tokens": 5000,
    },
}


@dataclass
class RunStats:
    label: str
    task: str
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: int = 0
    elapsed_s: float = 0.0
    response_chars: int = 0
    response: str = ""
    finish_reason: str = ""
    truncated: bool = False
    error: str | None = None
    axor_tracked_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def estimated_cost(self) -> float:
        cin, cout = COSTS_PER_M.get(self.provider, COSTS_PER_M["anthropic"])
        return self.input_tokens / 1_000_000 * cin + self.output_tokens / 1_000_000 * cout


@dataclass
class JudgeStats:
    score: float = -1.0
    verdict: str = ""
    accuracy: float = 0.0
    completeness: float = 0.0
    coherence: float = 0.0
    reasoning: str = ""
    error: str | None = None


def _large_block(title: str, body: str, *, target_kb: int) -> str:
    target = max(1, target_kb) * 1024
    chunks = [f"# {title}\n", body.strip(), "\n"]
    i = 1
    while len("\n".join(chunks)) < target:
        chunks.append(
            f"\n## Evidence slice {i}\n"
            f"{body.strip()}\n"
            f"Observation {i}: latency_delta_ms={320 + i * 17}; "
            f"error_rate_pct={round(0.2 + (i % 9) * 0.37, 2)}; "
            f"affected_tenants={12 + i % 41}; confidence=high.\n"
        )
        i += 1
    return "\n".join(chunks)[: target + 512]


def build_tools(tool_kb: int):
    from langchain.tools import tool

    @tool
    def list_repo_tree(area: str = "all") -> str:
        """Return a repository tree and ownership map for the requested area."""
        body = """
services/
  checkout-api/       owners: payments-platform, runtime: python/fastapi
  billing-worker/     owners: billing-core, runtime: python/celery
  identity-gateway/   owners: security-platform, runtime: go
  agent-orchestrator/ owners: ai-platform, runtime: python/langgraph
packages/
  authlib_ext/        token validators, API-key compatibility layer
  observability/      tracing, metrics, structured logging
  shared_db/          SQLAlchemy models, migrations, query helpers
Risk notes:
- checkout-api imports authlib_ext.legacy_api_keys in three request paths.
- billing-worker runs high-cardinality tenant aggregation every minute.
- agent-orchestrator stores complete message state in graph checkpoints.
"""
        return _large_block("Repository tree", body, target_kb=tool_kb)

    @tool
    def read_source_file(path: str) -> str:
        """Read a source file with security notes and nearby context."""
        body = f"""
File: {path}

def validate_request(req, tenant_id, token):
    claims = auth.validate(token)
    if claims.kind == "api_key":
        security_note = "legacy API keys bypass token audience validation"
        return LegacyPrincipal(tenant_id=req.headers["X-Tenant-ID"])
    if claims.aud != f"tenant:{{tenant_id}}":
        raise PermissionDenied("audience mismatch")
    return Principal(subject=claims.sub, tenant_id=tenant_id)

def checkout(payload, principal):
    regression_note = "inventory lookup moved before idempotency check"
    inventory = inventory_client.reserve(payload.items)
    order = idempotency.get_or_create(payload.key, payload)
    return payment_gateway.charge(order, inventory)

Known issues:
- Missing deadline propagation to inventory_client.reserve.
- No circuit breaker around payment_gateway.charge.
- API-key path accepts tenant id from header.
"""
        return _large_block(f"Source {path}", body, target_kb=tool_kb)

    @tool
    def grep_code(pattern: str) -> str:
        """Search the codebase and return matching files with snippets."""
        body = f"""
Search pattern: {pattern}

services/checkout-api/app/routes/checkout.py:47
  inventory = inventory_client.reserve(payload.items)
  order = idempotency.get_or_create(payload.key, payload)

packages/authlib_ext/legacy_api_keys.py:88
  tenant_id = request.headers.get("X-Tenant-ID")
  return LegacyPrincipal(tenant_id=tenant_id, scopes=["*"])

services/agent-orchestrator/state.py:133
  checkpoint.messages = prior_messages + tool_results
  note = "summarize old tool results before checkpoint writes"

services/billing-worker/jobs/tenant_rollup.py:212
  rows = db.execute("select * from events where tenant_id = :tenant")
"""
        return _large_block("Code search results", body, target_kb=tool_kb)

    @tool
    def query_metrics(metric: str, window: str = "24h") -> str:
        """Query service metrics and return time-series summaries."""
        body = f"""
Metric: {metric}
Window: {window}

checkout-api p95 latency:
  09:00 418ms
  09:15 431ms
  09:30 4720ms  <-- release 2026.04.17 deployed to 50%
  09:45 4810ms
  10:00 4920ms
  10:15 790ms   <-- rollback inventory change

DB query count per checkout:
  baseline=3.1, incident_peak=18.7, after_rollback=3.8

agent-orchestrator input tokens per run:
  p50=18k, p95=112k, p99=189k. Largest growth comes from tool outputs
  retained in graph state across researcher -> writer handoff.
"""
        return _large_block("Metrics query", body, target_kb=tool_kb)

    @tool
    def fetch_trace(trace_id: str) -> str:
        """Fetch a distributed trace with spans, timings, and annotations."""
        body = f"""
Trace: {trace_id}

root checkout POST /v1/checkout duration=4872ms status=200
  auth.validate duration=41ms
  idempotency.lookup duration=32ms
  inventory.reserve duration=4211ms retries=3
    inventory.db.select_stock duration=1180ms retry=0
    inventory.db.select_stock duration=1217ms retry=1
    inventory.db.select_stock duration=1266ms retry=2
  payment.charge duration=391ms
  emit.order_created duration=28ms

Annotations:
- retry reason: DEADLINE_EXCEEDED from inventory.
- no request deadline propagated from checkout-api.
- idempotency check happens after inventory reservation in new release.
"""
        return _large_block("Distributed trace", body, target_kb=tool_kb)

    @tool
    def read_incident_log(service: str, window: str = "2h") -> str:
        """Read structured logs for a service over a time window."""
        body = f"""
Service: {service}
Window: {window}

2026-04-17T09:28:10Z deploy_started version=checkout-api@8f91c2 canary=50%
2026-04-17T09:31:44Z warn inventory.reserve slow duration_ms=3880 retry=1
2026-04-17T09:33:02Z warn db_pool saturation pool=inventory_ro in_use=96/100
2026-04-17T09:39:50Z error checkout timeout tenant=enterprise-77 attempt=2
2026-04-17T09:49:12Z rollback_started reason="latency regression"
2026-04-17T10:13:04Z rollback_complete version=checkout-api@7b02aa

Correlated finding: traffic volume was normal. Regression aligns with release.
"""
        return _large_block("Incident logs", body, target_kb=tool_kb)

    @tool
    def inspect_runbook(topic: str) -> str:
        """Return operational runbook content for incident response or rollout."""
        body = f"""
Runbook: {topic}

Severity criteria:
- SEV1 if checkout p95 > 2s for more than 10 minutes or error rate > 5%.
- Page payments-platform primary and inventory on-call.

Mitigation:
1. Freeze deploys.
2. Reduce canary to 0% or roll back last checkout-api release.
3. Enable inventory circuit breaker if DB pool saturation remains > 80%.
4. Disable API-key compatibility path only after security signoff.

Post-incident:
- RCA due in 48h.
- Add regression tests for retry/idempotency ordering.
- Add SLO burn alert for db_query_count_per_checkout.
"""
        return _large_block("Runbook", body, target_kb=tool_kb)

    @tool
    def dependency_audit(package: str = "all") -> str:
        """Audit dependency versions, vulnerabilities, and migration blockers."""
        body = f"""
Dependency audit: {package}

critical:
- pyjwt 2.4.0 used in authlib_ext. Upgrade to >= 2.8.0 for stricter JWT
  validation defaults and security fixes.
- requests 2.28 in billing-worker lacks standard retry budget instrumentation.

high:
- sqlalchemy session lifecycle differs between checkout-api and billing-worker.
- langgraph checkpoint payload can exceed 8MB for research workflows.

medium:
- fastapi-users integration pinned to pre-OAuth migration adapter.

Migration blocker:
- enterprise tenants still use API keys for nightly billing export jobs.
"""
        return _large_block("Dependency audit", body, target_kb=tool_kb)

    @tool
    def read_design_doc(doc: str) -> str:
        """Read an architecture or migration design document."""
        body = f"""
Design doc: {doc}

Goals:
- Replace long-lived API keys with short-lived OAuth access tokens.
- Preserve enterprise integrations through a 90-day compatibility bridge.
- Make tenant binding explicit in token audience and subject claims.

Non-goals:
- No global auth rewrite.
- No breaking billing export jobs without tenant opt-in.

Rollout:
phase 0 inventory all API-key callers
phase 1 dual-write token issue telemetry
phase 2 require OAuth for interactive traffic
phase 3 rotate API keys to read-only export scopes
phase 4 delete legacy_api_keys.py

Backout:
- Keep validator dual-stack behind feature flag auth.oauth_required.
"""
        return _large_block("Design document", body, target_kb=tool_kb)

    @tool
    def security_scan(target: str) -> str:
        """Return security findings for a target service or package."""
        body = f"""
Security scan target: {target}

Finding SEC-1007 HIGH:
Legacy API key path trusts X-Tenant-ID header. A leaked API key with broad
scope could be replayed against another tenant if gateway routing is bypassed.

Finding SEC-1031 MEDIUM:
JWT refresh grace period set to zero, causing brittle migrations and forcing
operators to extend API-key compatibility longer than planned.

Finding SEC-1088 MEDIUM:
Audit logs omit token audience on failed validation attempts.

Recommended fixes:
- Bind tenant to credential server-side.
- Emit token kind, audience, subject, and validator path in auth audit logs.
"""
        return _large_block("Security scan", body, target_kb=tool_kb)

    @tool
    def query_ticket_history(query: str) -> str:
        """Search tickets, incidents, and postmortems."""
        body = f"""
Ticket query: {query}

INC-2026-0417 checkout latency regression:
- root cause suspected: inventory reservation before idempotency.
- mitigated by rollback after 42 minutes.

SEC-2026-0172 API key tenant spoofing risk:
- accepted risk until OAuth migration phase 2.
- enterprise migration blocked by billing export jobs.

COST-2026-0088 agent workflow cost spike:
- large tool outputs retained across graph nodes.
- writer node received full researcher logs and traces.
"""
        return _large_block("Ticket history", body, target_kb=tool_kb)

    @tool
    def run_test_suite(suite: str) -> str:
        """Run or retrieve a test suite summary with failures and logs."""
        body = f"""
Test suite: {suite}

Passed: 842
Failed: 3
Skipped: 17

Failures:
1. test_checkout_idempotency_before_inventory
   Expected idempotency lookup before inventory.reserve. Actual order reversed.
2. test_legacy_api_key_tenant_binding
   Expected tenant from credential metadata. Actual tenant from X-Tenant-ID.
3. test_agent_checkpoint_compaction
   Expected checkpoint < 4MB. Actual checkpoint 11.8MB after 6 tool calls.

Slow tests:
- test_inventory_retry_budget 41.2s
- test_oauth_dual_stack_rollout 29.8s
"""
        return _large_block("Test results", body, target_kb=tool_kb)

    @tool
    def load_customer_impact(segment: str = "all") -> str:
        """Load customer impact analysis by segment or tenant cohort."""
        body = f"""
Customer segment: {segment}

Impact:
- 18.4% of checkout requests exceeded 2s during the incident window.
- 2.1% exceeded 10s and were abandoned client-side.
- 37 enterprise tenants affected; 9 high-value tenants opened tickets.
- No confirmed duplicate charges, but 114 duplicate inventory holds.

Revenue:
- Estimated lost GMV: $184,000
- Support cost: 87 tickets, 31 escalations

Not affected:
- Read-only catalog browsing
- Subscription renewals
- Billing export jobs
"""
        return _large_block("Customer impact", body, target_kb=tool_kb)

    @tool
    def estimate_costs(workflow: str) -> str:
        """Estimate model, infra, and operational costs for a workflow."""
        body = f"""
Workflow: {workflow}

Current model cost:
- p50 run: 18k input / 1.2k output tokens
- p95 run: 112k input / 4.8k output tokens
- p99 run: 189k input / 6.1k output tokens

Cost drivers:
1. Tool outputs retained verbatim in graph state.
2. Writer receives researcher raw logs instead of condensed findings.
3. Repeated search queries return overlapping content.
4. No per-tool output cap or freshness policy.

Projected savings:
- fragment-aware compression: 25-45% on p95 runs.
- opt-in tool result cache for read-only doc tools: 5-12%.
- prompt cache for stable system/runbook context: 8-18%.
"""
        return _large_block("Cost estimate", body, target_kb=tool_kb)

    @tool
    def write_decision_matrix(topic: str) -> str:
        """Create a decision matrix for options, risks, and recommendation."""
        body = f"""
Decision matrix: {topic}

Option A: rollback only
- speed: high
- risk reduction: medium
- residual risk: regression can recur

Option B: rollback + regression tests + deadline propagation
- speed: medium
- risk reduction: high
- residual risk: inventory DB saturation still possible

Option C: broader architecture rewrite
- speed: low
- risk reduction: high
- residual risk: migration complexity

Recommendation: choose option B immediately, then schedule focused
architecture work for retry budgets and circuit breakers.
"""
        return _large_block("Decision matrix", body, target_kb=max(1, tool_kb // 2))

    return [
        list_repo_tree,
        read_source_file,
        grep_code,
        query_metrics,
        fetch_trace,
        read_incident_log,
        inspect_runbook,
        dependency_audit,
        read_design_doc,
        security_scan,
        query_ticket_history,
        run_test_suite,
        load_customer_impact,
        estimate_costs,
        write_decision_matrix,
    ]


def build_model(provider: str, model_id: str | None, max_tokens: int):
    if provider == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit("ANTHROPIC_API_KEY is required for --provider anthropic")
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise SystemExit(
                "Missing provider dependency: install `langchain-anthropic` "
                "or `pip install -e .[anthropic]`."
            ) from exc
        model = model_id or "claude-sonnet-4-6"
        return ChatAnthropic(model=model, max_tokens=max_tokens, temperature=0.0), model
    if provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY is required for --provider openai")
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise SystemExit(
                "Missing provider dependency: install `langchain-openai` "
                "or `pip install -e .[openai]`."
            ) from exc
        model = model_id or "gpt-4.1-mini"
        return ChatOpenAI(model=model, max_tokens=max_tokens, temperature=0.0), model
    raise SystemExit(f"unsupported provider: {provider}")


def build_prior_history(task_name: str, prior_turns: int, tool_kb: int) -> list[Any]:
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    history: list[Any] = []
    topics = [
        "checkout-api incident timeline",
        "inventory retry behavior",
        "OAuth migration blockers",
        "legacy API-key tenant binding",
        "agent checkpoint token growth",
        "billing export compatibility",
        "customer impact summary",
        "dependency audit notes",
        "test failure summary",
        "deployment runbook excerpts",
        "cost optimization history",
        "security exception review",
    ]
    for i, topic in enumerate(topics[:prior_turns]):
        tc_id = f"prior_{task_name}_{i}"
        history.append(HumanMessage(content=f"Earlier investigation request: {topic}"))
        history.append(AIMessage(
            content="",
            tool_calls=[{
                "name": "query_ticket_history",
                "args": {"query": topic},
                "id": tc_id,
            }],
        ))
        history.append(ToolMessage(
            content=_large_block(
                f"Prior tool output {i}",
                (
                    f"Prior evidence for {topic}. This output was kept in the "
                    "conversation state by an earlier production agent turn. "
                    "It includes logs, findings, repeated excerpts, and raw "
                    "diagnostic material that is useful but too large to keep "
                    "verbatim forever."
                ),
                target_kb=tool_kb,
            ),
            tool_call_id=tc_id,
        ))
        history.append(AIMessage(
            content=(
                f"Finding {i}: I chose to keep {topic} as relevant evidence. "
                "The next step should cross-check this against metrics, source "
                "code, tests, and runbooks."
            )
        ))
    return history


def estimate_tokens_text(text: str) -> int:
    return max(len(text) // 4, 1)


def estimate_tokens_messages(messages: list[Any]) -> int:
    return sum(estimate_tokens_text(_text_of(m)) for m in messages)


def estimate_tools_schema_tokens(tool_count: int) -> int:
    # Conservative schema estimate without coupling to LangChain internals.
    return tool_count * 320


def system_prompt(min_tool_calls: int) -> str:
    return (
        "You are a senior production engineering agent. You are operating on a "
        "high-stakes task and must ground your answer in tool evidence. Before "
        f"finalizing, use {min_tool_calls} distinct relevant tools when available: "
        "repo/source search, metrics, traces, logs, tickets, tests, runbooks, "
        "security/dependency analysis, impact, and cost. Do not paste raw tool "
        "outputs back to the user. Produce a concise but complete report with "
        "evidence bullets, risks, and prioritized actions."
    )


def make_agent(provider: str, model_id: str | None, max_tokens: int, *,
               task_name: str,
               with_axor: bool, tool_kb: int, min_tool_calls: int,
               recent_tools_window: int, compression_mode: str | None,
               optimization_profile: str | None):
    from langchain.agents import create_agent

    model, resolved_model = build_model(provider, model_id, max_tokens)
    tools = build_tools(tool_kb)
    middleware = []
    axor = None
    if with_axor:
        from axor_langchain import AxorMiddleware
        # Tool relevance gating, dedup, etc. are derived from
        # `optimization_profile`. No hand-curated per-task allowlist —
        # savings have to come from middleware logic, not the benchmark.
        axor = AxorMiddleware(
            soft_token_limit=250_000,
            hard_token_limit=350_000,
            tool_max_retries=1,
            tool_retry_delay=0.2,
            tool_error_handler=lambda name, exc: f"[tool {name} failed: {exc}]",
            track_tool_stats=True,
            verbose=False,
            optimization_profile=optimization_profile,
            recent_tools_window=recent_tools_window,
            compression_mode=compression_mode,
        )
        middleware.append(axor)

    kwargs: dict[str, Any] = {
        "tools": tools,
        "system_prompt": system_prompt(min_tool_calls),
    }
    if middleware:
        kwargs["middleware"] = middleware
    return create_agent(model, **kwargs), resolved_model, axor


def _text_of(msg: Any) -> str:
    content = getattr(msg, "content", "") or ""
    if isinstance(content, list):
        return " ".join(
            block.get("text", "") for block in content if isinstance(block, dict)
        )
    return str(content)


def _finish_reason_of(msg: Any) -> str:
    metadata = getattr(msg, "response_metadata", None) or {}
    if not isinstance(metadata, dict):
        metadata = {}
    for key in ("stop_reason", "finish_reason", "stop"):
        value = metadata.get(key)
        if value:
            return str(value)
    additional = getattr(msg, "additional_kwargs", None) or {}
    if isinstance(additional, dict):
        for key in ("stop_reason", "finish_reason", "stop"):
            value = additional.get(key)
            if value:
                return str(value)
    return ""


def _usage_from_messages(messages: list[Any]) -> tuple[int, int, int]:
    input_tokens = 0
    output_tokens = 0
    tool_calls = 0
    for msg in messages:
        if getattr(msg, "type", None) != "ai":
            continue
        tool_calls += len(getattr(msg, "tool_calls", None) or [])
        usage = getattr(msg, "usage_metadata", None) or {}
        if isinstance(usage, dict):
            input_tokens += usage.get("input_tokens", 0) or 0
            output_tokens += usage.get("output_tokens", 0) or 0
        else:
            input_tokens += getattr(usage, "input_tokens", 0) or 0
            output_tokens += getattr(usage, "output_tokens", 0) or 0
    return input_tokens, output_tokens, tool_calls


def _effective_attr(args: argparse.Namespace, name: str) -> Any:
    return getattr(args, f"effective_{name}", getattr(args, name))


def configure_effective_payload(args: argparse.Namespace) -> None:
    """Shrink synthetic payloads enough for common live API TPM limits."""
    args.effective_prior_turns = args.prior_turns
    args.effective_prior_kb = args.prior_kb
    args.effective_tool_kb = args.tool_kb
    args.effective_min_tool_calls = args.min_tool_calls
    args.effective_sleep_between = (
        args.sleep_between
        if args.sleep_between is not None
        else (65.0 if args.provider == "anthropic" else 2.0)
    )

    if args.no_auto_fit:
        return

    # Keep initial Anthropic calls below common 30k input TPM limits.
    tpm = args.input_tpm_limit or (30_000 if args.provider == "anthropic" else 100_000)
    target_initial = args.max_initial_input_tokens or int(tpm * 0.45)
    target_initial = max(4_000, target_initial)

    if args.provider == "anthropic" and tpm <= 30_000:
        args.effective_tool_kb = min(args.effective_tool_kb, 3)
        args.effective_min_tool_calls = min(args.effective_min_tool_calls, 6)

    largest_task = max(TASKS, key=lambda name: len(TASKS[name]))
    while True:
        prior = build_prior_history(
            largest_task,
            args.effective_prior_turns,
            args.effective_prior_kb,
        )
        from langchain_core.messages import HumanMessage
        prior.append(HumanMessage(content=TASKS[largest_task]))
        estimated = (
            estimate_tokens_text(system_prompt(args.effective_min_tool_calls))
            + estimate_tokens_messages(prior)
            + estimate_tools_schema_tokens(15)
            + 1_500
        )
        args.estimated_initial_input_tokens = estimated
        if estimated <= target_initial:
            break
        if args.effective_prior_kb > 1:
            args.effective_prior_kb -= 1
        elif args.effective_prior_turns > 2:
            args.effective_prior_turns -= 1
        elif args.effective_tool_kb > 1:
            args.effective_tool_kb -= 1
        else:
            break


def run_once(args: argparse.Namespace, task_name: str, *, with_axor: bool) -> RunStats:
    from langchain_core.messages import HumanMessage

    label = "governed" if with_axor else "baseline"
    prompt = TASKS[task_name]
    agent, model, axor = make_agent(
        args.provider,
        args.model,
        args.max_tokens,
        task_name=task_name,
        with_axor=with_axor,
        tool_kb=_effective_attr(args, "tool_kb"),
        min_tool_calls=_effective_attr(args, "min_tool_calls"),
        recent_tools_window=args.recent_tools_window,
        compression_mode=args.compression_mode,
        optimization_profile=(
            None if args.axor_profile == "custom" else args.axor_profile
        ),
    )
    messages = build_prior_history(
        task_name,
        _effective_attr(args, "prior_turns"),
        _effective_attr(args, "prior_kb"),
    )
    messages.append(HumanMessage(content=prompt))

    stats = RunStats(label=label, task=task_name, provider=args.provider, model=model)
    t0 = time.monotonic()
    try:
        result = agent.invoke(
            {"messages": messages},
            config={"recursion_limit": args.recursion_limit},
        )
        stats.elapsed_s = time.monotonic() - t0
        out_messages = result.get("messages", [])
        stats.input_tokens, stats.output_tokens, stats.tool_calls = _usage_from_messages(out_messages)
        ai_messages = [
            m for m in out_messages
            if getattr(m, "type", None) == "ai" and _text_of(m)
        ]
        stats.response = _text_of(ai_messages[-1]) if ai_messages else ""
        stats.finish_reason = _finish_reason_of(ai_messages[-1]) if ai_messages else ""
        stats.truncated = stats.finish_reason in {"max_tokens", "length"}
        stats.response_chars = len(stats.response)
        if axor is not None:
            stats.axor_tracked_tokens = axor.total_tokens_spent
    except Exception as exc:
        stats.elapsed_s = time.monotonic() - t0
        stats.error = f"{type(exc).__name__}: {exc}"
    return stats


def build_judge(args: argparse.Namespace):
    if not args.judge:
        return None
    provider = args.judge_provider or args.provider
    if provider == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit("ANTHROPIC_API_KEY is required for Anthropic judge")
        from axor_langchain.judge import make_anthropic_judge
        return make_anthropic_judge(model=args.judge_model or "claude-sonnet-4-6")
    if provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY is required for OpenAI judge")
        from axor_langchain.judge import make_openai_judge
        try:
            return make_openai_judge(model=args.judge_model or "gpt-4.1-mini")
        except ImportError as exc:
            raise SystemExit(
                "Missing judge dependency: install `langchain-openai` "
                "or `pip install -e .[openai]`."
            ) from exc
    raise SystemExit(f"unsupported judge provider: {provider}")


def run_judge(args: argparse.Namespace, task_name: str,
              baseline: RunStats, governed: RunStats, judge_llm) -> JudgeStats:
    stats = JudgeStats()
    if judge_llm is None or baseline.error or governed.error:
        return stats
    if args.judge_sleep > 0:
        print(f"  sleeping {args.judge_sleep:.0f}s before judge...", flush=True)
        time.sleep(args.judge_sleep)
    print(f"[{task_name}] judging...", flush=True)
    try:
        from axor_langchain.judge import quality_judge
        result = quality_judge(
            task=TASKS[task_name],
            baseline_response=baseline.response,
            governed_response=governed.response,
            llm=judge_llm,
            response_char_limit=args.judge_response_chars,
        )
        stats = JudgeStats(
            score=result.score,
            verdict=result.verdict,
            accuracy=result.accuracy,
            completeness=result.completeness,
            coherence=result.coherence,
            reasoning=result.reasoning,
        )
        print(
            f"  JUDGE: score={stats.score:.2f} verdict={stats.verdict} "
            f"accuracy={stats.accuracy:.2f} completeness={stats.completeness:.2f} "
            f"coherence={stats.coherence:.2f}"
        )
        if stats.reasoning:
            print(f"  JUDGE reasoning: {stats.reasoning[:500]}")
    except Exception as exc:
        stats.error = f"{type(exc).__name__}: {exc}"
        print(f"  JUDGE ERROR: {stats.error}")
    return stats


def compare(args: argparse.Namespace, task_name: str,
            judge_llm=None) -> tuple[RunStats, RunStats, JudgeStats]:
    print(f"\n[{task_name}] baseline...", flush=True)
    baseline = run_once(args, task_name, with_axor=False)
    print_run(baseline)
    if args.effective_sleep_between > 0:
        print(f"  sleeping {args.effective_sleep_between:.0f}s for rate limit window...", flush=True)
        time.sleep(args.effective_sleep_between)
    print(f"[{task_name}] governed...", flush=True)
    governed = run_once(args, task_name, with_axor=True)
    print_run(governed)
    judge = run_judge(args, task_name, baseline, governed, judge_llm)
    return baseline, governed, judge


def print_run(run: RunStats) -> None:
    if run.error:
        print(f"  ERROR after {run.elapsed_s:.1f}s: {run.error}")
        return
    print(
        f"  {run.input_tokens:,} in / {run.output_tokens:,} out "
        f"({run.total_tokens:,} total), tools={run.tool_calls}, "
        f"response={run.response_chars:,} chars, {run.elapsed_s:.1f}s, "
        f"est_cost=${run.estimated_cost():.4f}"
    )
    if run.truncated:
        print(f"  WARNING: final response stopped by {run.finish_reason}; increase --max-tokens")


def print_comparison(baseline: RunStats, governed: RunStats) -> None:
    if baseline.error or governed.error:
        return
    token_delta = baseline.total_tokens - governed.total_tokens
    input_delta = baseline.input_tokens - governed.input_tokens
    cost_delta = baseline.estimated_cost() - governed.estimated_cost()
    token_savings = token_delta / baseline.total_tokens * 100 if baseline.total_tokens else 0.0
    input_savings = input_delta / baseline.input_tokens * 100 if baseline.input_tokens else 0.0
    cost_savings = cost_delta / baseline.estimated_cost() * 100 if baseline.estimated_cost() else 0.0
    print(
        "  SAVINGS: "
        f"input={input_savings:+.1f}% "
        f"total={token_savings:+.1f}% "
        f"cost={cost_savings:+.1f}% "
        f"delta_tokens={token_delta:+,} "
        f"delta_cost=${cost_delta:+.4f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live hard benchmark for AxorMiddleware")
    parser.add_argument("--axor-profile",
                        choices=["cautious", "aggressive", "custom"],
                        default="aggressive",
                        help="named benchmark preset; aggressive is the validated hard-agent default")
    parser.add_argument("--provider", choices=["anthropic", "openai"], default="anthropic")
    parser.add_argument("--model", default=None)
    parser.add_argument("--task", choices=[*TASKS.keys(), "all"], default="incident_rca")
    parser.add_argument("--prior-turns", type=int, default=8,
                        help="number of valid prior tool-result turns to preload")
    parser.add_argument("--prior-kb", type=int, default=8,
                        help="KB per prior ToolMessage")
    parser.add_argument("--tool-kb", type=int, default=8,
                        help="KB returned by most live tools")
    parser.add_argument("--min-tool-calls", type=int, default=8,
                        help="tool-use target embedded in the system prompt")
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--recent-tools-window", type=int, default=None,
                        help="recent ToolMessages Axor keeps verbatim; try 1 for more savings")
    parser.add_argument("--compression-mode",
                        choices=["auto", "aggressive", "balanced", "light"],
                        default=None,
                        help="override Axor policy-selected compression mode")
    parser.add_argument("--max-savings", action="store_true",
                        help="shortcut: window 0 + aggressive compression")
    parser.add_argument("--recursion-limit", type=int, default=40)
    parser.add_argument("--sleep-between", type=float, default=None,
                        help="seconds between live API runs; default 65 for Anthropic, 2 otherwise")
    parser.add_argument("--input-tpm-limit", type=int, default=None,
                        help="provider input-token/minute limit used for auto-fit")
    parser.add_argument("--max-initial-input-tokens", type=int, default=None,
                        help="preflight target for the first model call")
    parser.add_argument("--no-auto-fit", action="store_true",
                        help="do not shrink synthetic payloads for live rate limits")
    parser.add_argument("--judge", action="store_true",
                        help="run LLM-as-judge on baseline vs governed responses")
    parser.add_argument("--judge-provider", choices=["anthropic", "openai"], default=None,
                        help="judge provider; defaults to --provider")
    parser.add_argument("--judge-model", default=None,
                        help="judge model id; provider default when omitted")
    parser.add_argument("--judge-sleep", type=float, default=65.0,
                        help="seconds to sleep before judge call")
    parser.add_argument("--judge-response-chars", type=int, default=20_000,
                        help="max chars per response sent to judge; head+tail fit if exceeded")
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON at the end")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.axor_profile != "custom":
        profile_cfg = BENCHMARK_AXOR_PROFILES[args.axor_profile]
        if args.max_tokens is None:
            args.max_tokens = profile_cfg["max_tokens"]
        if args.recent_tools_window is None:
            args.recent_tools_window = profile_cfg["recent_tools_window"]
        if args.compression_mode is None:
            args.compression_mode = profile_cfg["compression_mode"]
    else:
        if args.max_tokens is None:
            args.max_tokens = 1800
        if args.recent_tools_window is None:
            args.recent_tools_window = 2
        if args.compression_mode is None:
            args.compression_mode = "auto"
    if args.max_savings:
        args.recent_tools_window = 0
        args.compression_mode = "aggressive"
    configure_effective_payload(args)
    task_names = list(TASKS) if args.task == "all" else [args.task]
    judge_llm = build_judge(args)
    all_results: list[dict[str, Any]] = []

    print("Axor live hard-agent benchmark")
    print(
        f"provider={args.provider} model={args.model or 'default'} "
        f"axor_profile={args.axor_profile}"
    )
    print(
        f"prior_turns={args.prior_turns} prior_kb={args.prior_kb} "
        f"tool_kb={args.tool_kb} recent_tools_window={args.recent_tools_window} "
        f"compression_mode={args.compression_mode} "
        f"recursion_limit={args.recursion_limit}"
    )
    if not args.no_auto_fit:
        print(
            "auto_fit="
            f"prior_turns={args.effective_prior_turns} "
            f"prior_kb={args.effective_prior_kb} "
            f"tool_kb={args.effective_tool_kb} "
            f"min_tool_calls={args.effective_min_tool_calls} "
            f"estimated_initial_input={args.estimated_initial_input_tokens:,} "
            f"sleep_between={args.effective_sleep_between:.0f}s"
        )

    for idx, task_name in enumerate(task_names):
        baseline, governed, judge = compare(args, task_name, judge_llm)
        print_comparison(baseline, governed)
        all_results.append({
            "task": task_name,
            "baseline": asdict(baseline),
            "governed": asdict(governed),
            "judge": asdict(judge),
        })
        if idx < len(task_names) - 1 and args.effective_sleep_between > 0:
            print(f"sleeping {args.effective_sleep_between:.0f}s before next task...", flush=True)
            time.sleep(args.effective_sleep_between)

    if args.json:
        print("\nJSON")
        print(json.dumps(all_results, indent=2))

    failed = any(r["baseline"]["error"] or r["governed"]["error"] for r in all_results)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
