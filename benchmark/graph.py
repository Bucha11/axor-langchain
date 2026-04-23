"""
Benchmark graph — message builders and content generators.

Provides:
  - Real LangChain messages (HumanMessage, AIMessage, ToolMessage)
  - Proper tool_calls / tool_call_id pairing
  - Realistic prior conversation history
  - Pre-scripted model responses for FakeMessagesListChatModel
  - Tool definitions for agent pipeline
"""

from __future__ import annotations

import uuid
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool


# ── Token estimation ──────────────────────────────────────────────────────────

def estimate_tokens(text: str) -> int:
    """Approximate token count. ~4 chars/token for English mixed content.
    Real tokenizers (BPE) differ by 15-30% — use for relative comparison only."""
    return max(len(text) // 4, 1)


# ── Realistic content generators ──────────────────────────────────────────────

def make_tool_output(topic: str, n: int) -> str:
    return f"""Search results for '{topic}' (result {n}):

Title: Comprehensive Analysis of {topic}
URL: https://example.com/research/{topic.replace(' ', '-')}/{n}
Published: 2025-03-{n + 1:02d}

Abstract: This paper presents a detailed examination of {topic} from multiple
perspectives. The authors conducted extensive research involving 847 data points
collected over 18 months from 12 different geographic regions. Statistical
analysis using ANOVA (p < 0.001) confirmed the hypothesis. Key findings include:

1. Primary finding: {topic} demonstrates a 34.7% improvement in efficiency
   when governance controls are applied systematically across all pipeline stages.

2. Secondary finding: Token accumulation in multi-agent pipelines follows an
   exponential growth pattern without compression, reaching O(n²) complexity
   by turn 8-10. This was observed consistently across 47 different pipeline
   configurations tested in the study.

3. Tertiary finding: Memory compression with semantic prioritization reduced
   context window usage by 67-82% while maintaining 94.3% output quality as
   measured by human evaluators on a 5-point Likert scale.

Methodology: Mixed-methods approach combining quantitative token analysis with
qualitative assessment of output coherence. Full dataset available upon request.
Reference: DOI:10.1234/example.{n}.2025

Related papers: [23 citations omitted for brevity]
Tags: machine-learning, agent-systems, token-optimization, governance"""


def make_plan(task: str) -> str:
    return f"""I'll break down '{task}' into three research subtasks:

1. **Literature review**: Search for recent papers and studies on the topic.
   - Focus on peer-reviewed sources from 2023-2025
   - Identify key researchers and institutions in the field
   - Note contradictory findings for balanced analysis

2. **Data analysis**: Examine quantitative evidence and benchmarks.
   - Collect numerical data points and statistics
   - Compare methodologies across studies
   - Identify gaps in current research

3. **Synthesis**: Combine findings into actionable insights.
   - Cross-reference sources for consistency
   - Identify the 3-5 most significant findings
   - Frame conclusions for the target audience

Estimated scope: 8-12 sources, 2,000-3,000 word final report.
Priority: accuracy over speed. Flag any contradictory findings."""


def make_research(n_results: int) -> str:
    return f"""Research complete. Analyzed {n_results} sources.

**Key findings:**

1. Strong consensus across sources that systematic approaches outperform
   ad-hoc methods by 40-60% on standardized benchmarks.

2. Counter-finding: Three studies suggest governance overhead can reduce
   throughput by 8-15% in latency-sensitive single-node architectures.

3. Emerging trend: Hybrid approaches combining rule-based and ML-based
   selection show most promise at 78.3% performance / 23% of the cost.

**Source quality assessment:**
- High quality (peer-reviewed): 8 sources
- Medium quality (industry reports): 3 sources
- Low quality (excluded): 1 source

**Gaps identified:** No longitudinal studies beyond 6 months."""


def make_final_report(task: str, research: str) -> str:
    return f"""# Research Report: {task}

## Executive Summary

Based on comprehensive analysis of 12 peer-reviewed sources and industry
reports, this report presents findings on {task}.

## Key Findings

{research}

## Recommendations

1. **Immediate**: Implement context compression in all multi-agent pipelines.
   Expected token reduction: 50-75% based on pipeline depth.

2. **Short-term (1-3 months)**: Evaluate hybrid governance approaches.

3. **Long-term (6-12 months)**: Invest in longitudinal studies.

## Conclusion

ROI analysis suggests 3-4x cost reduction is achievable within 90 days."""


def make_analysis_output(dataset: str, metric: str) -> str:
    return f"""Analysis of '{dataset}' on metric '{metric}':

Statistical summary:
  n = 1,247 observations  |  Period: Jan 2024 – Apr 2025
  Mean:    42.3     |  Median: 38.7
  Std dev: 12.4     |  IQR:    [31.2, 52.8]

Trend: +18.4% YoY (p < 0.001, ANOVA)
Outliers: 23 detected via IQR method, removed from main analysis.

Segmentation:
  Group A (n=623): mean=48.1, significantly higher (p=0.003)
  Group B (n=624): mean=36.5
  Effect size: Cohen's d = 0.72 (large effect)

Conclusion: Strong signal in the data supporting the research hypothesis."""


# ── Tool definitions ──────────────────────────────────────────────────────────

@tool
def search_web(query: str) -> str:
    """Search the web for recent information on a topic."""
    return make_tool_output(query, 1)


@tool
def analyze_data(dataset: str, metric: str) -> str:
    """Analyze a dataset and return key statistics for a given metric."""
    return make_analysis_output(dataset, metric)


@tool
def summarize_findings(findings: str, max_words: int = 200) -> str:
    """Summarize research findings into a concise executive summary."""
    word_count = min(len(findings.split()), max_words)
    return f"""Executive Summary ({word_count} words):

The research presents compelling evidence for systematic approaches.
Three key themes emerge: EFFICIENCY (40-60% gains), COST (50-70% reduction),
QUALITY (94%+ maintained at 67-82% token reduction).
Confidence level: HIGH. Recommended action: implement immediately."""


ALL_TOOLS = [search_web, analyze_data, summarize_findings]
TOOL_NAMES = [t.name for t in ALL_TOOLS]


# ── Prior history builder ─────────────────────────────────────────────────────

def _tc_id() -> str:
    return f"tc_{uuid.uuid4().hex[:8]}"


SUBTOPICS = [
    "background on {task}",
    "recent developments in {task}",
    "key challenges of {task}",
    "cost implications of {task}",
    "best practices for {task}",
    "case studies of {task}",
    "future outlook for {task}",
    "technical details of {task}",
]

ASSISTANT_REPLIES = [
    "Based on my research, the evidence strongly supports systematic approaches. "
    "I found 5 high-quality sources with consistent findings across methodologies. "
    "The primary conclusion is that governance controls yield 40-60% efficiency gains.",

    "The data analysis reveals clear trends. Statistical significance is high (p<0.001). "
    "I recommend focusing on the top 3 interventions based on effect size.",

    "My research plan: (1) literature review, (2) data analysis, (3) synthesis. "
    "I'll prioritize peer-reviewed sources and cross-validate findings.",
]

TOOL_SUMMARIES = [
    "Found 5 relevant papers. Key finding: 34.7% efficiency gain with governance.",
    "Analysis complete: n=1247, mean=42.3, strong YoY trend (+18.4%).",
    "Summarized 8 sources. Three themes: efficiency, cost, quality.",
    "Search returned 4 high-quality industry reports from 2024-2025.",
    "Data shows bimodal distribution. Group A significantly outperforms Group B.",
]


def build_prior_history(task: str, n_turns: int) -> list:
    """
    Build realistic prior conversation with proper tool-call pairing.

    Each turn: HumanMessage → AIMessage(tool_calls) → ToolMessage(tool_call_id) → AIMessage(reply)
    This is the real-world pattern that causes token explosion.
    """
    history = []
    for i in range(min(n_turns, len(SUBTOPICS))):
        subtopic = SUBTOPICS[i].format(task=task)
        tc = _tc_id()

        history.append(HumanMessage(content=f"Research subtopic: {subtopic}"))
        history.append(AIMessage(
            content="",
            tool_calls=[{"name": "search_web", "args": {"query": subtopic}, "id": tc}],
        ))
        history.append(ToolMessage(
            content=TOOL_SUMMARIES[i % len(TOOL_SUMMARIES)],
            tool_call_id=tc,
        ))
        history.append(AIMessage(
            content=ASSISTANT_REPLIES[i % len(ASSISTANT_REPLIES)],
        ))
    return history


# ── Pre-scripted model responses for FakeMessagesListChatModel ────────────────

def build_planner_responses(task: str) -> list[AIMessage]:
    """Two responses: (1) tool call to search_web, (2) final plan."""
    tc = _tc_id()
    return [
        AIMessage(
            content="",
            tool_calls=[{"name": "search_web", "args": {"query": task}, "id": tc}],
        ),
        AIMessage(content=make_plan(task)),
    ]


def build_researcher_responses(task: str) -> list[AIMessage]:
    """Three responses: (1) search_web call, (2) analyze_data call, (3) findings."""
    tc1, tc2 = _tc_id(), _tc_id()
    return [
        AIMessage(
            content="",
            tool_calls=[{"name": "search_web", "args": {"query": task}, "id": tc1}],
        ),
        AIMessage(
            content="",
            tool_calls=[{"name": "analyze_data", "args": {"dataset": task, "metric": "efficiency"}, "id": tc2}],
        ),
        AIMessage(content=make_research(n_results=2)),
    ]


def build_writer_responses(task: str) -> list[AIMessage]:
    """Two responses: (1) summarize_findings call, (2) final report."""
    tc = _tc_id()
    research = make_research(n_results=2)
    return [
        AIMessage(
            content="",
            tool_calls=[{"name": "summarize_findings", "args": {"findings": research[:300]}, "id": tc}],
        ),
        AIMessage(content=make_final_report(task, research)),
    ]


def count_messages_tokens(messages: list) -> tuple[int, int]:
    """Count (n_messages, estimated_tokens) in a message list."""
    n = len(messages)
    tokens = sum(estimate_tokens(getattr(m, "content", "") or "") for m in messages)
    return n, tokens
