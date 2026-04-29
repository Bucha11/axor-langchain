from __future__ import annotations

from axor_langchain.judge import _fit_text_for_judge, quality_judge


def test_fit_text_for_judge_preserves_tail_when_truncated():
    text = "A" * 100 + "MIDDLE" + "Z" * 100
    fitted = _fit_text_for_judge(text, 80)
    assert "judge input truncated" in fitted
    assert fitted.startswith("A")
    assert fitted.endswith("Z")
    assert "MIDDLE" not in fitted


def test_fit_text_for_judge_does_not_inflate_when_limit_below_marker():
    # Regression: previously `text[-0:]` returned the whole string when the
    # caller picked a limit shorter than the truncation marker, so the
    # "truncated" output was longer than the input.
    text = "A" * 500
    fitted = _fit_text_for_judge(text, 50)
    assert "judge input truncated" in fitted
    assert "A" * 500 not in fitted


def test_quality_judge_uses_full_default_limit_for_long_reports():
    baseline = "baseline " * 1200 + "BASELINE_END"
    governed = "governed " * 1200 + "GOVERNED_END"
    seen = {}

    def fake_llm(prompt: str) -> str:
        seen["prompt"] = prompt
        return (
            '{"accuracy":1.0,"completeness":1.0,"coherence":1.0,'
            '"score":1.0,"verdict":"equivalent","reasoning":"ok"}'
        )

    result = quality_judge(
        "task",
        baseline,
        governed,
        llm=fake_llm,
    )

    assert result.score == 1.0
    assert "BASELINE_END" in seen["prompt"]
    assert "GOVERNED_END" in seen["prompt"]


def test_quality_judge_handles_curly_braces_in_responses():
    """Regression: agent responses containing JSON/code with raw `{`/`}` must
    not crash the judge. Previously str.format raised KeyError on `{x}`.
    """
    baseline = '{"hello": "world", "set": {1, 2, 3}}'
    governed = "def f(x): return {x: x*2 for x in range(10)}"
    seen = {}

    def fake_llm(prompt: str) -> str:
        seen["prompt"] = prompt
        return '{"score":0.95,"verdict":"equivalent","reasoning":"ok"}'

    result = quality_judge("task with {curly}", baseline, governed, llm=fake_llm)
    assert result.score == 0.95
    assert "{curly}" in seen["prompt"]
    assert baseline in seen["prompt"]
    assert governed in seen["prompt"]
