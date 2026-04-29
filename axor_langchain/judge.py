from __future__ import annotations

"""
LLM-as-judge: compare a governed agent response to a baseline response
for the same task. Returns a quality score in [0, 1] + reasoning.

Without this, claims like "−30% cost without quality loss" are unfalsifiable.
The judge gives a regression net: a verdict of `regression` (score < 0.70)
indicates the compression policy broke something material; `minor_drift`
(0.70 ≤ score < 0.90) is usually acceptable; `equivalent` (score ≥ 0.90)
means the governed run is interchangeable with the baseline.

Design notes:
  • Sync API. Middleware code path is sync; we run judge AFTER the agent
    finishes, not during.
  • Provider-agnostic: caller passes a thin LLM-call callable. Tests
    inject a mock; production wires it via `make_anthropic_judge` /
    `make_openai_judge`.
  • Defensive JSON parsing — judge models prepend prose, wrap in fences,
    sometimes return malformed JSON. Best-effort extraction, never crashes.
  • Byte-identical short-circuit: skips the LLM call when responses are
    equal, returns score=1.0. Saves judge tokens on trivial cases.
"""

import json
import re
from dataclasses import dataclass
from typing import Callable


@dataclass
class JudgeResult:
    score: float
    verdict: str
    reasoning: str
    accuracy: float = 0.0
    completeness: float = 0.0
    coherence: float = 0.0
    raw: str = ""


JudgeLLM = Callable[[str], str]


_JUDGE_PROMPT = """\
You are evaluating whether a "governed" agent response is equivalent in
quality to a "baseline" agent response for the SAME task.

Both responses came from the same agent on the same task; the only
difference is that the governed run had prompt compression / context
deduplication / older-message truncation applied. The question is
whether the cost savings broke quality.

Score on three axes (0.0 = totally broken, 1.0 = matches baseline):

1. ACCURACY: does the governed response claim the same facts as the
   baseline? Penalize hallucinations or factual contradictions vs baseline.
2. COMPLETENESS: does it cover the same scope? Penalize missing
   important points the baseline covered.
3. COHERENCE: is it well-formed independently? Penalize broken
   structure, abrupt cutoffs, or confusion suggesting context loss.

Final score = simple average of the three axes.

Verdict labels:
  "equivalent"   — score ≥ 0.90, governed is interchangeable with baseline
  "minor_drift"  — 0.70 ≤ score < 0.90, savings probably worth the cost
  "regression"   — score < 0.70, compression broke something material

──────────────────────────────────────────────────────────────────────
TASK:
__TASK__

BASELINE RESPONSE:
__BASELINE__

GOVERNED RESPONSE:
__GOVERNED__
──────────────────────────────────────────────────────────────────────

Return ONLY a JSON object, no prose before or after, no markdown fences:
{
  "accuracy": <float>,
  "completeness": <float>,
  "coherence": <float>,
  "score": <float>,
  "verdict": "<equivalent|minor_drift|regression>",
  "reasoning": "<one paragraph>"
}
"""


def _fit_text_for_judge(text: str, limit: int) -> str:
    """Fit long responses without creating a fake abrupt ending.

    Prefix-only truncation made complete long reports look cut off to the
    judge. Keep head and tail so the model can see both the main analysis and
    whether the response finished cleanly.
    """
    text = text or ""
    if limit <= 0 or len(text) <= limit:
        return text
    marker = (
        "\n\n[... judge input truncated: middle omitted; "
        f"original_chars={len(text)} ...]\n\n"
    )
    remaining = max(limit - len(marker), 0)
    head = remaining // 2
    tail = remaining - head
    # text[-0:] returns the whole string in Python — guard against it.
    return text[:head] + marker + (text[-tail:] if tail > 0 else "")


def _parse_judge_output(text: str) -> JudgeResult:
    """
    Real models prepend prose, wrap in ```json fences, occasionally emit
    invalid JSON. We extract the first {...} block we find and json.loads it.
    On any failure we return a regression verdict so the bench fails loudly
    rather than silently averaging in 0.0.
    """
    raw = text or ""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        candidate = match.group(0) if match else ""

    if not candidate:
        return JudgeResult(
            score=0.0, verdict="regression",
            reasoning=f"judge output unparseable: {raw[:200]!r}",
            raw=raw,
        )

    try:
        data = json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        return JudgeResult(
            score=0.0, verdict="regression",
            reasoning=f"judge JSON malformed: {candidate[:200]!r}",
            raw=raw,
        )

    score = max(0.0, min(1.0, float(data.get("score", 0.0))))
    verdict = str(data.get("verdict", "regression"))
    if verdict not in ("equivalent", "minor_drift", "regression"):
        verdict = "regression"
    return JudgeResult(
        score=score,
        verdict=verdict,
        reasoning=str(data.get("reasoning", ""))[:1000],
        accuracy=float(data.get("accuracy", 0.0)),
        completeness=float(data.get("completeness", 0.0)),
        coherence=float(data.get("coherence", 0.0)),
        raw=raw,
    )

def quality_judge(
    task: str,
    baseline_response: str,
    governed_response: str,
    *,
    llm: JudgeLLM,
    response_char_limit: int = 20_000,
) -> JudgeResult:
    """
    Score the governed response against the baseline via the supplied LLM.

    `llm(prompt) -> text` is called exactly once per non-trivial pair.
    Caller is responsible for retries, rate limiting, and provider choice.

    Optimization: byte-identical responses skip the LLM entirely
    (returns 1.0). Saves judge tokens on trivial / cached cases.
    """
    if not (baseline_response or governed_response):
        return JudgeResult(
            score=1.0, verdict="equivalent",
            reasoning="both responses empty — nothing to judge",
        )
    if baseline_response.strip() == governed_response.strip():
        return JudgeResult(
            score=1.0, verdict="equivalent",
            accuracy=1.0, completeness=1.0, coherence=1.0,
            reasoning="byte-identical to baseline",
        )

    # Use literal-token replacement instead of str.format — agent outputs
    # can contain raw curly braces (JSON, code, sets) which would crash
    # str.format with KeyError/IndexError. Tokens are unambiguous markers.
    prompt = (
        _JUDGE_PROMPT
        .replace("__TASK__", task[:4000])
        .replace("__BASELINE__", _fit_text_for_judge(baseline_response, response_char_limit))
        .replace("__GOVERNED__", _fit_text_for_judge(governed_response, response_char_limit))
    )
    output = llm(prompt)
    return _parse_judge_output(output)

def make_anthropic_judge(model: str = "claude-sonnet-4-6") -> JudgeLLM:
    """Build a judge callable via langchain-anthropic."""
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError as exc:
        raise ImportError(
            "langchain-anthropic is required for Anthropic judging; "
            "install `axor-langchain[anthropic]`."
        ) from exc
    chat = ChatAnthropic(model=model, max_tokens=1024, temperature=0.0)

    def _llm(prompt: str) -> str:
        msg = chat.invoke(prompt)
        content = getattr(msg, "content", str(msg))
        if isinstance(content, list):
            return " ".join(
                b.get("text", "") for b in content if isinstance(b, dict)
            )
        return str(content)

    return _llm


def make_openai_judge(model: str = "gpt-4.1-mini") -> JudgeLLM:
    """Build a judge callable via langchain-openai."""
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise ImportError(
            "langchain-openai is required for OpenAI judging; "
            "install `axor-langchain[openai]`."
        ) from exc
    chat = ChatOpenAI(model=model, max_tokens=1024, temperature=0.0)

    def _llm(prompt: str) -> str:
        msg = chat.invoke(prompt)
        content = getattr(msg, "content", str(msg))
        if isinstance(content, list):
            return " ".join(
                b.get("text", "") for b in content if isinstance(b, dict)
            )
        return str(content)

    return _llm
