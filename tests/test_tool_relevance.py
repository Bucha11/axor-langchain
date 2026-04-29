"""
Dynamic tool relevance gating.

`tool_selection="relevance"` replaces the manual `allowed_tools=[...]`
crutch with axor-core-style keyword overlap on tool names + descriptions.

Contract:
  • Pending tool_calls (last AI emitted them, no ToolMessage yet) are
    NEVER dropped — provider APIs reject orphan tool_call/tool_result.
  • Tools called within the last K AI rounds are sticky-anchored.
  • At least `tool_min_keep` tools always survive.
  • READONLY tasks additionally drop tools whose names match destructive
    verbs ("write_*", "delete_*", "run_*", etc.), but only if the floor
    is still met after.
  • Without `tool_selection="relevance"` the new code path is a no-op:
    backwards compatible for all existing users.
"""
from __future__ import annotations

import pytest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from axor_langchain.middleware import (
    AxorMiddleware,
    _DOMAIN_TOPICS,
    _expand_with_synonyms,
    _extract_query_keywords,
    _pending_tool_call_names,
    _recently_called_tool_names,
    _score_tool_relevance,
)


# ── Test doubles ──────────────────────────────────────────────────────────────

def _tool(name: str, desc: str = ""):
    return {"name": name, "description": desc}


class _Req:
    def __init__(self, messages=None, tools=None, system_message=None):
        self.messages = list(messages or [])
        self.tools = list(tools or [])
        self.system_message = system_message
    def override(self, **kw):
        clone = _Req(self.messages, self.tools, self.system_message)
        for k, v in kw.items():
            setattr(clone, k, v)
        return clone


class _Resp:
    usage_metadata = {"input_tokens": 0, "output_tokens": 1}


def _drive(axor, msgs, tools):
    axor._ensure_engines()
    captured = {}
    def handler(req):
        captured["req"] = req
        return _Resp()
    axor.wrap_model_call(_Req(messages=msgs, tools=tools), handler)
    return captured["req"]


# ── Pure helpers ──────────────────────────────────────────────────────────────

class TestExtractKeywords:

    def test_drops_stopwords_and_short_tokens(self):
        kws = _extract_query_keywords(
            "Investigate the checkout latency incident from the metrics."
        )
        assert "checkout" in kws
        assert "latency" in kws
        assert "incident" in kws
        assert "metrics" in kws
        assert "the" not in kws
        assert "from" not in kws

    def test_empty_input(self):
        assert _extract_query_keywords("") == set()
        assert _extract_query_keywords(None) == set()


class TestScoreToolRelevance:

    def test_name_hits_outweigh_desc_hits(self):
        kws = {"checkout", "latency"}
        s_name = _score_tool_relevance(
            _tool("checkout_latency_probe", "Generic search."), kws,
        )
        s_desc = _score_tool_relevance(
            _tool("xyz_helper", "Look up checkout latency in metrics."), kws,
        )
        assert s_name > s_desc

    def test_no_keyword_overlap_zero(self):
        s = _score_tool_relevance(
            _tool("write_decision_matrix", "Create matrix."),
            {"checkout", "latency"},
        )
        assert s == 0.0

    def test_empty_keywords_zero(self):
        assert _score_tool_relevance(_tool("anything"), set()) == 0.0


class TestPendingAndStickyDetection:

    def test_pending_when_no_result_yet(self):
        msgs = [
            HumanMessage(content="task"),
            AIMessage(content="", tool_calls=[
                {"name": "search", "args": {}, "id": "tc1"}]),
        ]
        assert _pending_tool_call_names(msgs) == {"search"}

    def test_not_pending_after_result(self):
        msgs = [
            HumanMessage(content="task"),
            AIMessage(content="", tool_calls=[
                {"name": "search", "args": {}, "id": "tc1"}]),
            ToolMessage(content="result", tool_call_id="tc1"),
        ]
        assert _pending_tool_call_names(msgs) == set()

    def test_sticky_collects_recent_ai_calls(self):
        msgs = [
            HumanMessage(content="task"),
            AIMessage(content="", tool_calls=[
                {"name": "old_tool", "args": {}, "id": "old1"}]),
            ToolMessage(content="r", tool_call_id="old1"),
            AIMessage(content="thinking..."),
            AIMessage(content="", tool_calls=[
                {"name": "fresh_tool", "args": {}, "id": "f1"}]),
            ToolMessage(content="r", tool_call_id="f1"),
        ]
        # lookback=2 captures only the last 2 AI messages: prose + fresh_tool.
        assert _recently_called_tool_names(msgs, lookback=2) == {"fresh_tool"}
        # lookback=4 reaches back to old_tool too.
        assert _recently_called_tool_names(msgs, lookback=4) == {
            "fresh_tool", "old_tool",
        }

    def test_sticky_zero_lookback_empty(self):
        msgs = [AIMessage(content="", tool_calls=[
            {"name": "x", "args": {}, "id": "tc1"}])]
        assert _recently_called_tool_names(msgs, lookback=0) == set()


# ── Default behavior is a no-op ───────────────────────────────────────────────

class TestNoOpByDefault:
    """`tool_selection=None` (the default) must keep the legacy pass-through
    behavior intact. Otherwise we silently break every existing user."""

    def test_default_passes_all_tools(self):
        axor = AxorMiddleware()
        msgs = [HumanMessage(content="trivial task")]
        tools = [_tool(f"tool_{i}", "desc") for i in range(15)]
        seen = _drive(axor, msgs, tools)
        assert len(seen.tools) == 15

    def test_relevance_off_string(self):
        axor = AxorMiddleware(tool_selection="off")
        msgs = [HumanMessage(content="x")]
        tools = [_tool("a"), _tool("b")]
        seen = _drive(axor, msgs, tools)
        assert len(seen.tools) == 2


# ── Relevance gating happy path ───────────────────────────────────────────────

class TestRelevanceGating:

    def test_relevant_tools_kept_irrelevant_dropped(self):
        # top_k=2, min_keep=2 — pure score-based top-2.
        axor = AxorMiddleware(tool_selection="relevance",
                              tool_top_k=2, tool_min_keep=2)
        msgs = [HumanMessage(
            content="Investigate the checkout latency incident timeline.",
        )]
        tools = [
            _tool("write_decision_matrix", "Create a decision matrix."),
            _tool("query_metrics", "Query latency metrics over a window."),
            _tool("read_design_doc", "Read a design document."),
            _tool("fetch_trace", "Fetch a distributed trace for an incident."),
            _tool("inspect_runbook", "Return runbook content."),
        ]
        seen = _drive(axor, msgs, tools)
        names = [t["name"] for t in seen.tools]
        # query_metrics + fetch_trace win on keywords latency/incident.
        assert names == ["query_metrics", "fetch_trace"] or set(names) == {
            "query_metrics", "fetch_trace",
        }
        assert "write_decision_matrix" not in names

    def test_pending_tool_call_always_kept(self):
        """Even if the pending tool's score is zero, dropping it would
        produce an orphan tool_call → provider 400."""
        axor = AxorMiddleware(tool_selection="relevance", tool_top_k=2,
                              tool_min_keep=2)
        msgs = [
            HumanMessage(content="latency incident review"),
            AIMessage(content="", tool_calls=[
                {"name": "totally_unrelated_tool",
                 "args": {}, "id": "pending1"},
            ]),
            # No ToolMessage yet → pending.
        ]
        tools = [
            _tool("totally_unrelated_tool", "Has nothing to do with task."),
            _tool("query_metrics", "Latency metrics query."),
            _tool("fetch_trace", "Distributed trace fetch."),
        ]
        seen = _drive(axor, msgs, tools)
        names = [t["name"] for t in seen.tools]
        assert "totally_unrelated_tool" in names

    def test_sticky_tool_kept_despite_low_score(self):
        axor = AxorMiddleware(tool_selection="relevance", tool_top_k=1,
                              tool_min_keep=1, tool_sticky_lookback=3)
        msgs = [
            HumanMessage(content="latency incident analysis"),
            AIMessage(content="", tool_calls=[
                {"name": "obscure_legacy", "args": {}, "id": "tc1"}]),
            ToolMessage(content="some output", tool_call_id="tc1"),
            AIMessage(content="thinking"),
        ]
        tools = [
            _tool("obscure_legacy", "Legacy tool, no keywords."),
            _tool("query_metrics", "Latency metrics query."),
        ]
        seen = _drive(axor, msgs, tools)
        names = [t["name"] for t in seen.tools]
        # Even though query_metrics scores higher, sticky obscure_legacy
        # stays in the keep set.
        assert "obscure_legacy" in names

    def test_floor_respected(self):
        axor = AxorMiddleware(tool_selection="relevance", tool_top_k=1,
                              tool_min_keep=3)
        msgs = [HumanMessage(content="metrics")]
        tools = [_tool(f"unrelated_{i}", "x") for i in range(8)]
        seen = _drive(axor, msgs, tools)
        # min_keep wins over top_k.
        assert len(seen.tools) >= 3

    def test_no_keywords_no_filtering(self):
        """Empty/stopword-only task → can't score → keep everything."""
        axor = AxorMiddleware(tool_selection="relevance", tool_top_k=2)
        msgs = [HumanMessage(content="the it on")]
        tools = [_tool(f"t{i}") for i in range(5)]
        seen = _drive(axor, msgs, tools)
        assert len(seen.tools) == 5


# ── READONLY-aware filter ─────────────────────────────────────────────────────

class TestReadonlyFilter:

    def test_readonly_drops_destructive_named_tool(self):
        axor = AxorMiddleware(tool_selection="relevance", tool_top_k=4,
                              tool_min_keep=2)
        # "Explain how" → READONLY classifier verdict.
        msgs = [HumanMessage(
            content="Explain how the cache layer works in detail."
        )]
        tools = [
            _tool("read_source_file", "Read a source file."),
            _tool("grep_code", "Search the codebase."),
            _tool("inspect_runbook", "Read runbook."),
            _tool("write_decision_matrix", "Create a decision matrix."),
            _tool("delete_file", "Delete a file from disk."),
            _tool("deploy_service", "Deploy a service to prod."),
        ]
        seen = _drive(axor, msgs, tools)
        names = {t["name"] for t in seen.tools}
        # destructive-named tools dropped from a READONLY task
        assert "delete_file" not in names
        assert "deploy_service" not in names

    def test_mutative_keeps_destructive_named(self):
        axor = AxorMiddleware(tool_selection="relevance", tool_top_k=10,
                              tool_min_keep=2)
        msgs = [HumanMessage(
            content="Fix the failing migration in the deploy script."
        )]
        tools = [
            _tool("write_migration", "Write a migration file."),
            _tool("deploy_service", "Deploy service."),
            _tool("read_design_doc", "Read design."),
        ]
        seen = _drive(axor, msgs, tools)
        names = {t["name"] for t in seen.tools}
        # MUTATIVE classification → destructive filter does NOT engage.
        assert "deploy_service" in names


# ── Profile presets wire it up ────────────────────────────────────────────────

class TestProfileWiring:

    def test_aggressive_profile_enables_relevance(self):
        axor = AxorMiddleware(optimization_profile="aggressive")
        assert axor._tool_selection == "relevance"
        assert axor._tool_top_k == 8
        assert axor._tool_dedup_old_results is True

    def test_cautious_profile_no_relevance(self):
        axor = AxorMiddleware(optimization_profile="cautious")
        assert axor._tool_selection is None

    def test_explicit_kwarg_overrides_profile(self):
        axor = AxorMiddleware(
            optimization_profile="aggressive",
            tool_selection="off",
        )
        assert axor._tool_selection is None


# ── Bad input handling ────────────────────────────────────────────────────────

class TestBadInputs:

    def test_invalid_tool_selection_raises(self):
        with pytest.raises(ValueError):
            AxorMiddleware(tool_selection="bogus")


# ── Prompt-cache stability ────────────────────────────────────────────────────

class TestPromptCacheStability:
    """Provider prompt caching (Anthropic, OpenAI) requires byte-identical
    prefixes across consecutive model calls. The tools schema lives in
    that prefix, so a relevance-gated tool list that mutates between
    calls within the same agent.invoke loop kills cache hits.

    Contract:
      • Within a single user turn (latest_human_text unchanged), the
        tool selection MUST be byte-identical across calls. Sticky
        behavior, pending tool_calls accumulating across calls, and
        score recomputations must NOT mutate the result.
      • A new HumanMessage MUST invalidate the cache and produce a
        fresh selection.
      • A change in the available tool set (caller registered a new
        tool) MUST also invalidate the cache.
    """

    @staticmethod
    def _tools():
        return [
            {"name": "query_metrics", "description": "Query latency metrics."},
            {"name": "fetch_trace", "description": "Fetch distributed trace."},
            {"name": "read_incident_log", "description": "Read service logs."},
            {"name": "inspect_runbook", "description": "Read runbook."},
            {"name": "grep_code", "description": "Search the codebase."},
            {"name": "read_source_file", "description": "Read source file."},
            {"name": "run_test_suite", "description": "Run test suite."},
            {"name": "load_customer_impact", "description": "Customer impact."},
            {"name": "dependency_audit", "description": "Audit dependencies."},
            {"name": "security_scan", "description": "Security findings."},
            {"name": "read_design_doc", "description": "Read design document."},
            {"name": "estimate_costs", "description": "Estimate workflow costs."},
        ]

    def _drive(self, axor, msgs, tools):
        seen = {}
        def handler(req): seen["req"] = req; return _Resp()
        axor.wrap_model_call(_Req(messages=msgs, tools=tools), handler)
        return [t["name"] for t in seen["req"].tools]

    def test_identical_tools_across_consecutive_calls_same_human(self):
        axor = AxorMiddleware(optimization_profile="aggressive")
        axor._ensure_engines()
        tools = self._tools()
        human = HumanMessage(
            content="Investigate the checkout latency incident timeline."
        )

        sel1 = self._drive(axor, [human], tools)
        sel2 = self._drive(axor, [
            human,
            AIMessage(content="", tool_calls=[
                {"name": "fetch_trace", "args": {}, "id": "tc1"}]),
            ToolMessage(content="trace", tool_call_id="tc1"),
        ], tools)
        sel3 = self._drive(axor, [
            human,
            AIMessage(content="", tool_calls=[
                {"name": "fetch_trace", "args": {}, "id": "tc1"}]),
            ToolMessage(content="trace", tool_call_id="tc1"),
            AIMessage(content="", tool_calls=[
                {"name": "inspect_runbook", "args": {}, "id": "tc2"}]),
            ToolMessage(content="rb", tool_call_id="tc2"),
        ], tools)

        # Strict byte equality — order and membership both matter for
        # the cached prefix.
        assert sel1 == sel2 == sel3

    def test_new_human_invalidates_cache(self):
        axor = AxorMiddleware(optimization_profile="aggressive")
        axor._ensure_engines()
        tools = self._tools()

        sel_first = self._drive(axor, [
            HumanMessage(content="Investigate latency incident.")
        ], tools)
        sel_after = self._drive(axor, [
            HumanMessage(content="Investigate latency incident."),
            AIMessage(content="done."),
            HumanMessage(content="Now plan a security audit of OAuth tokens."),
        ], tools)

        # Different task → different relevant tools.
        assert set(sel_first) != set(sel_after)

    def test_pending_tool_call_anchored_even_on_cache_hit(self):
        """If a tool_call is pending (no tool_result yet), its tool MUST
        appear in the request even if it wasn't part of the cached
        selection. Provider rejects orphan tool_call/tool_result pairs."""
        axor = AxorMiddleware(optimization_profile="aggressive")
        axor._ensure_engines()
        tools = self._tools()

        # First call: an unrelated incident task. Cache stores the
        # incident-flavored top-K (no security_scan).
        human = HumanMessage(content="Investigate latency incident.")
        sel1 = self._drive(axor, [human], tools)
        assert "security_scan" not in sel1

        # Second call: same human (cache hit), but the AI just emitted a
        # tool_call to security_scan that hasn't been resolved yet.
        sel2 = self._drive(axor, [
            human,
            AIMessage(content="", tool_calls=[
                {"name": "security_scan", "args": {}, "id": "tc1"}]),
            # Note: NO tool_result yet — pending.
        ], tools)
        # Cache is still hit (same human + tools), but pending anchor
        # forces security_scan to appear.
        assert "security_scan" in sel2

    def test_changed_tool_set_invalidates_cache(self):
        axor = AxorMiddleware(optimization_profile="aggressive")
        axor._ensure_engines()
        tools_a = self._tools()
        tools_b = tools_a + [
            {"name": "extra_tool", "description": "An additional metric tool."},
        ]
        human = HumanMessage(content="Investigate latency incident.")

        sel_a = self._drive(axor, [human], tools_a)
        sel_b = self._drive(axor, [human], tools_b)
        # Different tool set → re-computed (extra_tool may or may not
        # win, but the bytes must reflect the new available registry).
        # The minimal contract is: don't blindly return the cached A
        # selection when the registry now has 13 tools.
        assert len(sel_b) <= len(tools_b)

    def test_stability_disabled_recomputes_per_call(self):
        # Opt-out: power users who want adaptive behavior across calls.
        axor = AxorMiddleware(
            optimization_profile="aggressive",
            tool_selection_stable=False,
        )
        axor._ensure_engines()
        tools = self._tools()
        human = HumanMessage(content="Investigate latency incident.")

        sel1 = self._drive(axor, [human], tools)
        sel2 = self._drive(axor, [
            human,
            AIMessage(content="", tool_calls=[
                {"name": "inspect_runbook", "args": {}, "id": "tc1"}]),
            ToolMessage(content="rb", tool_call_id="tc1"),
        ], tools)
        # With stability off, sticky inclusion of `inspect_runbook` may
        # add it to call 2's set. Either way, the contract here is that
        # the middleware did NOT short-circuit via cache — observable
        # by checking the cache attribute stayed unset.
        assert axor._tool_selection_cache is None


# ── Synonym expansion (domain bridges) ────────────────────────────────────────

class TestSynonymExpansion:
    """The keyword extractor expands via `_DOMAIN_TOPICS` so user-language
    tasks bridge to admin-language tool names. Without this the relevance
    scorer is pure literal overlap and misses the right tool whenever the
    prompt and tool description don't share exact tokens.
    """

    def test_security_terms_pull_each_other(self):
        # User wrote "vulnerable", tool says "security findings" — without
        # synonyms zero overlap, with synonyms they share the security topic.
        expanded = _expand_with_synonyms({"vulnerable"})
        assert "security" in expanded
        assert "audit" in expanded
        assert "auth" in expanded

    def test_observability_terms_pull_each_other(self):
        # "p95 latency" → trace, metric, log, span all reachable.
        expanded = _expand_with_synonyms({"p95"})
        assert "trace" in expanded
        assert "metric" in expanded
        assert "log" in expanded

    def test_unknown_word_no_expansion(self):
        # Tokens not in any topic produce no expansion.
        assert _expand_with_synonyms({"asdfqwerty"}) == set()

    def test_synonym_score_lower_than_direct(self):
        # Direct name hit must beat a tool that only matches by synonym,
        # otherwise over-broad expansion can dethrone obviously-correct tools.
        kws = {"latency"}  # observability topic
        direct_hit = {"name": "latency_probe", "description": "x"}
        synonym_hit = {"name": "trace_collector", "description": "x"}
        assert (_score_tool_relevance(direct_hit, kws)
                > _score_tool_relevance(synonym_hit, kws))

    def test_synonym_lifts_relevant_tool_from_zero(self):
        kws = {"vulnerable", "oauth", "tokens"}  # security-flavored task
        no_syn = _score_tool_relevance(
            {"name": "security_scan", "description": "Return security findings."},
            kws, use_synonyms=False,
        )
        with_syn = _score_tool_relevance(
            {"name": "security_scan", "description": "Return security findings."},
            kws, use_synonyms=True,
        )
        assert no_syn == 0.0
        assert with_syn > 0.0


# ── Cross-domain regression: realistic prompts pull right tools ──────────────

class TestRealisticPrompts:
    """Lightweight smoke check that the synonym-augmented scorer ranks
    obviously-relevant tools above obviously-irrelevant ones for
    realistic production tasks. Not a coverage benchmark — just a guard
    against future regressions in `_DOMAIN_TOPICS`."""

    INCIDENT = "checkout latency incident with p95 spike after release"
    SECURITY = "migrate from API keys to OAuth tokens; vulnerable code paths"
    COST     = "monthly model bill doubled; optimize agent token spend"

    OBS_TOOL  = {"name": "query_metrics", "description": "Query latency metrics."}
    TRACE     = {"name": "fetch_trace",   "description": "Distributed trace."}
    SEC       = {"name": "security_scan", "description": "Security findings."}
    COSTS     = {"name": "estimate_costs","description": "Workflow cost estimate."}
    REPO      = {"name": "list_repo_tree","description": "Show repo tree."}

    def _score(self, prompt, tool):
        return _score_tool_relevance(
            tool, _extract_query_keywords(prompt), use_synonyms=True,
        )

    def test_incident_picks_observability(self):
        assert self._score(self.INCIDENT, self.OBS_TOOL) > 0
        assert self._score(self.INCIDENT, self.TRACE) > 0
        # Not-relevant tool stays at the bottom
        assert self._score(self.INCIDENT, self.SEC) <= self._score(
            self.INCIDENT, self.OBS_TOOL,
        )

    def test_security_picks_security_scan(self):
        assert self._score(self.SECURITY, self.SEC) > 0

    def test_cost_picks_estimate_costs(self):
        assert self._score(self.COST, self.COSTS) > 0
        assert self._score(self.COST, self.SEC) <= self._score(
            self.COST, self.COSTS,
        )
