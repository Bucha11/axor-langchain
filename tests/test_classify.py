"""
FragmentValue classification correctness.

`_classify_messages(messages)` is a pure function that labels each
message with an axor-core `FragmentValue` based on (type, age,
content). Output feeds `axor_core.context.ContextCompressor` via
`_messages_to_fragments`.

Mapping (must hold across these tests):

  PINNED     SystemMessage,
             every HumanMessage,
             latest AIMessage with tool_calls (the "current decision" anchor),
             recent K ToolMessages (the "fresh tool outputs" — never trim,
             v1 lesson learned)
  KNOWLEDGE  AI prose containing decision verbs ("decided", "chose", ...)
  WORKING    AI prose without decision content,
             older AIMessage with tool_calls
  EPHEMERAL  ToolMessages older than the recent-K window
"""
from __future__ import annotations

import pytest

from axor_core.contracts.memory import FragmentValue

from axor_langchain.middleware import (
    _classify_messages,
    _has_decision_content,
)


# ── Test doubles ──────────────────────────────────────────────────────────────

def _msg(type_: str, content: str = "", *, tool_calls=None):
    """Plain stub message — no model_copy etc., just type + content."""
    class _M:
        pass
    m = _M()
    m.type = type_
    m.content = content
    if tool_calls is not None:
        m.tool_calls = tool_calls
    return m


def _values(classified):
    """Strip messages; return just the FragmentValue list for easy asserts."""
    return [v for _, v in classified]


# ── _has_decision_content ─────────────────────────────────────────────────────

class TestHasDecisionContent:

    def test_decided(self):
        assert _has_decision_content("I decided to use Postgres for this.")

    def test_chose(self):
        assert _has_decision_content("Chose pytest over unittest because of fixtures.")

    def test_will_use(self):
        assert _has_decision_content("We will use ripgrep for the search step.")

    def test_replaced(self):
        assert _has_decision_content("Replaced the old auth module with JWT.")

    def test_renamed(self):
        # `renamed` is in axor-core's DECISION_VERBS_RE
        assert _has_decision_content("Renamed get_x to fetch_x for consistency.")

    def test_fixed(self):
        assert _has_decision_content("Fixed the off-by-one in the loop.")

    def test_no_match(self):
        assert not _has_decision_content("Just thinking out loud here.")

    def test_empty(self):
        assert not _has_decision_content("")
        assert not _has_decision_content(None)


# ── PINNED rules ──────────────────────────────────────────────────────────────

class TestPinned:

    def test_system_is_pinned(self):
        msgs = [_msg("system", "you are X"), _msg("human", "hi")]
        assert _values(_classify_messages(msgs))[0] == FragmentValue.PINNED

    def test_every_human_is_pinned(self):
        msgs = [
            _msg("human", "first task"),
            _msg("ai", "ack"),
            _msg("human", "follow up"),
        ]
        vals = _values(_classify_messages(msgs))
        assert vals[0] == FragmentValue.PINNED
        assert vals[2] == FragmentValue.PINNED

    def test_latest_ai_with_tool_calls_is_pinned(self):
        msgs = [
            _msg("system", "sys"),
            _msg("human", "task"),
            _msg("ai", "", tool_calls=[{"name": "search", "id": "tc1"}]),
            _msg("tool", "result"),
            _msg("ai", "", tool_calls=[{"name": "fetch", "id": "tc2"}]),
        ]
        vals = _values(_classify_messages(msgs))
        # The LAST AIMessage with tool_calls (idx 4) is PINNED
        assert vals[4] == FragmentValue.PINNED
        # The earlier one (idx 2) is WORKING (older decision)
        assert vals[2] == FragmentValue.WORKING


# ── KNOWLEDGE rules: AI prose with decision verbs ────────────────────────────

class TestKnowledge:

    def test_ai_prose_with_decision_is_knowledge(self):
        msgs = [
            _msg("human", "task"),
            _msg("ai", "I decided to use the cache here."),
        ]
        vals = _values(_classify_messages(msgs))
        assert vals[1] == FragmentValue.KNOWLEDGE

    def test_ai_prose_without_decision_is_working(self):
        msgs = [
            _msg("human", "task"),
            _msg("ai", "Let me think about this carefully and look around."),
        ]
        vals = _values(_classify_messages(msgs))
        assert vals[1] == FragmentValue.WORKING

    def test_ai_prose_no_tool_calls_attr(self):
        # AIMessage without `tool_calls` attribute defaults to no calls
        msgs = [_msg("human", "x"), _msg("ai", "Refactored the utils module.")]
        vals = _values(_classify_messages(msgs))
        assert vals[1] == FragmentValue.KNOWLEDGE  # has "refactored"


# ── Tool message age window ───────────────────────────────────────────────────

class TestToolWindow:

    def _build(self, n_tools: int):
        msgs = [_msg("system", "sys"), _msg("human", "task")]
        for i in range(n_tools):
            msgs.append(_msg("ai", "", tool_calls=[{"name": "x", "id": f"t{i}"}]))
            msgs.append(_msg("tool", f"result {i}"))
        return msgs

    def test_only_two_tools_both_pinned_default_window(self):
        msgs = self._build(2)
        vals = _values(_classify_messages(msgs))  # default window=2
        tool_vals = [v for m, v in zip(msgs, vals)
                     if getattr(m, "type", None) == "tool"]
        assert tool_vals == [FragmentValue.PINNED, FragmentValue.PINNED]

    def test_three_tools_oldest_is_ephemeral(self):
        msgs = self._build(3)
        vals = _values(_classify_messages(msgs))
        tool_vals = [v for m, v in zip(msgs, vals)
                     if getattr(m, "type", None) == "tool"]
        # tool[0] = EPHEMERAL (oldest, beyond window=2)
        # tool[1], tool[2] = PINNED (recent two — fresh outputs, never trim)
        assert tool_vals == [
            FragmentValue.EPHEMERAL,
            FragmentValue.PINNED,
            FragmentValue.PINNED,
        ]

    def test_window_zero_all_tools_ephemeral(self):
        msgs = self._build(3)
        vals = _values(_classify_messages(msgs, recent_tools_window=0))
        tool_vals = [v for m, v in zip(msgs, vals)
                     if getattr(m, "type", None) == "tool"]
        assert tool_vals == [FragmentValue.EPHEMERAL] * 3

    def test_window_larger_than_count_all_pinned(self):
        msgs = self._build(2)
        vals = _values(_classify_messages(msgs, recent_tools_window=10))
        tool_vals = [v for m, v in zip(msgs, vals)
                     if getattr(m, "type", None) == "tool"]
        assert tool_vals == [FragmentValue.PINNED, FragmentValue.PINNED]


# ── End-to-end realistic scenario ─────────────────────────────────────────────

class TestRealisticScenario:
    """Mirror a typical research flow: 1 system + 1 human + several
    tool rounds + final AI answer."""

    def test_research_flow_classification(self):
        msgs = [
            _msg("system", "you are a research agent"),
            _msg("human", "Research X"),
            _msg("ai", "", tool_calls=[{"name": "search", "id": "t1"}]),
            _msg("tool", "result 1: lots of data"),
            _msg("ai", "I chose to dig deeper into source 2."),  # decision
            _msg("ai", "", tool_calls=[{"name": "fetch", "id": "t2"}]),
            _msg("tool", "result 2: more data"),
            _msg("ai", "", tool_calls=[{"name": "search", "id": "t3"}]),
            _msg("tool", "result 3: final piece"),
            _msg("ai", "Final answer: foo bar baz"),  # no decision verbs
        ]
        vals = _values(_classify_messages(msgs))

        assert vals[0] == FragmentValue.PINNED      # system
        assert vals[1] == FragmentValue.PINNED      # human
        assert vals[2] == FragmentValue.WORKING     # old AI tool_calls
        assert vals[3] == FragmentValue.EPHEMERAL   # old tool result
        assert vals[4] == FragmentValue.KNOWLEDGE   # AI prose with decision
        assert vals[5] == FragmentValue.WORKING     # 2nd AI tool_calls (not latest)
        assert vals[6] == FragmentValue.PINNED      # tool 2 (recent → PINNED)
        assert vals[7] == FragmentValue.PINNED      # latest AI with tool_calls
        assert vals[8] == FragmentValue.PINNED      # tool 3 (recent → PINNED)
        assert vals[9] == FragmentValue.WORKING     # final prose, no decision

    def test_no_messages_returns_empty(self):
        assert _classify_messages([]) == []

    def test_only_system_and_human(self):
        msgs = [_msg("system", "sys"), _msg("human", "task")]
        vals = _values(_classify_messages(msgs))
        assert vals == [FragmentValue.PINNED, FragmentValue.PINNED]
