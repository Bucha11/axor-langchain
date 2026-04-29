# Changelog

## 0.4.0 — 2026-04-29

### Added
- `axor_langchain.judge` — LLM-as-judge for governed-vs-baseline response
  quality scoring, used to make claims like "−30% cost without quality
  loss" falsifiable. Public surface:
  `JudgeResult`, `quality_judge`, `make_anthropic_judge`,
  `make_openai_judge`. Verdicts: `equivalent` (≥0.90), `minor_drift`
  (0.70–0.90), `regression` (<0.70). Byte-identical short-circuit skips
  the judge call when responses match.
- `AxorCancelledError` and `AxorState` re-exported from
  `axor_langchain` for consumers handling cancellation and middleware
  state directly.
- `benchmark/live_hard_agent.py` — live hard-agent suite: real provider
  calls, large prior history, deterministic tools, optional LLM-as-judge,
  auto-fit to common input-TPM limits. Profiles: `cautious`, `aggressive`,
  `custom`.

### Changed
- `axor-core` moved from optional `[core]` extra to a required dependency.
  Pin: `axor-core>=0.4.0,<0.5` (was the optional extra `>=0.2.0`).
- New provider extras: `[anthropic]`, `[openai]`, `[providers]` —
  pull `langchain-anthropic` / `langchain-openai`.
- 177 tests pass (was 173 at 0.3.1).

### Removed
- Simulated `benchmark/run.py`, live `benchmark/live_graph.py`, and the
  intermediate `benchmark/graph.py`. The live hard-agent suite is now
  the only supported benchmark.
- `docs/demo.{cast,sh,svg}` — asciinema demo moved out of the package.

## 0.3.1 — 2026-04-24

### Added
- One-time stderr hint when `telemetry='local'/'remote'` is passed but
  `axor-telemetry` is not installed (`pip install axor-langchain[telemetry]`).
  Suppressible with `AXOR_NO_BANNER=1`.

## 0.3.0 — 2026-04-24

### Added
- `AxorMiddleware(telemetry='off'|'local'|'remote')` kwarg plus
  `AXOR_TELEMETRY` env. One-time stderr notice on first init when
  telemetry is off (`AXOR_NO_BANNER=1` mutes). Shared marker with
  `axor-cli` at `~/.axor/.telemetry_notice_shown`.
- `aafter_agent` classifies the latest human message via axor-core
  `TaskAnalyzer` and emits `AnonymizedTraceRecord` via
  `TelemetryPipeline.record_decision`.
- New optional extra: `[telemetry]` → `axor-telemetry>=0.1.0`.
- 13 helper tests + 7 telemetry integration tests.

## 0.2.0 — 2026-04-23

### Changed
- Pin minimum bumped to `axor-core>=0.2.0` so
  `pip install axor-langchain[core]` cannot pull a version missing
  `ContextFragment.turn`.
- CI/PyPI/Python/License badges added to README.
