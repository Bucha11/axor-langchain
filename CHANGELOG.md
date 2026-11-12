# Changelog

## 0.6.2 — 2026-11-11

### Added
- **`AxorMiddleware(plane=...)`** — Control-Plane governance for a LangChain
  agent. Pass an axor-wrap `PlaneAdmission`; every model call and tool call is a
  boundary where the plane is honoured:
  - **pause** holds the call until resume; **stop** cancels the run
    (`AxorCancelledError`, reason `operator_stop`);
  - **excision** — messages whose `id` / `tool_call_id` the operator names are
    redacted (not dropped, so tool_call pairing stays valid) on that and every
    later model call; the system prompt is reported as operator config, so an
    excision naming it is refused whole (needs axor-core >= 0.12.1);
  - **test-bench injection** goes into exactly one model call, marked untrusted,
    never written to agent state;
  - **operator call cap** bounds real tool executions (cache hits are free);
    over the cap raises `BudgetExceededError`.
  Sync agents need `PlaneAdmission.posture()` (axor-wrap >= 0.2.2).
- **Connects on its own.** `AxorMiddleware(plane="https://cp…", node_id=...)`,
  or no argument at all with `AXOR_PLANE_URL` / `AXOR_NODE_ID` /
  `AXOR_INGEST_KEY` set: a background thread keeps the node live on the plane
  (desired-state subscription + heartbeat) for sync and async agents alike, and
  delivers queued acks at `close()` / interpreter exit. `plane=False` opts out
  even with the env set; an env-driven plane without the extra logs a warning
  instead of breaking the agent. New extra: `axor-langchain[plane]`.
- **The plane sees the agent.** With a link, every invocation is a plane run:
  `tool_call` / `tool_result` lines whose `value_ref` is the `tool_call_id` —
  the ref an excision cuts by — with taint by locator (url → web, path → file)
  and `arg_refs` where an argument quotes an earlier result. With `[security]`:
  each run becomes core's `SessionAuditRecord`, goes through axor-sentinel's
  `CoreSessionSink` and evidence model, and the cross-run `ReputationSnapshot`
  is posted. `probe=True` runs an axor-probe battery after each run with the
  agent's own model and, on drift, `localize` over the tool results; the
  proposal lands in the plane's Health panel. Verified end to end against a
  local plane: drift → proposed cut → operator self-heal → cut in context →
  re-probe CONSISTENT → heal resolved.
- An excision whose targets are not in the current context waits (stays
  pending) instead of being refused: a LangChain invocation rebuilds its
  context, so a cut commanded between runs applies when the thread carries it.

### Fixed
- `awrap_tool_call` never checked `cancel()` — an async agent kept executing
  tools after cancellation.

## 0.6.1 — 2026-11-04

### Changed
- `IntentNormalizer` is imported from `axor_core.policy.normalizer`. It moved
  there in axor-core 0.9.0; this package imported `axor_core.node.normalizer`,
  which is where it lived up to 0.8.x. That import was correct for the pin this
  package declared (`<0.7`) — moving it is what lets the pin reach a current
  kernel at all.

### Constraints
- Pin: `axor-core>=0.9,<0.13` (was `>=0.6.0,<0.7`). The floor is the release
  where `axor_core.policy.normalizer` first exists, so it is a consequence of
  the import above, not a separate narrowing. The ceiling covers the 0.12 line;
  0.12's breaking changes are STRICT-only and this package builds no governor.
  Measured against an installed 0.11.0 and an installed 0.12.0 — 192 tests pass
  on each.

## 0.6.0 — 2026-05-25

Tagged and published (`v0.6.0`, 38e2eae) without changelog entries; recorded
here from the tag, not re-derived from the diff. It introduced
`axor_langchain.wrapper` and `axor_langchain.normalizer`, extended
`axor_langchain.middleware`, and added `_version.py` with its version test.

## 0.5.0 — 2026-05-04

### Added
- Per-run tool-selection telemetry. `_prepare_model_request` snapshots
  `mode` (`"none"`/`"relevance"`), `offered`, `kept`, `dropped_relevance`,
  and `dropped_denied`; `_record_telemetry` forwards the snapshot via
  the new `record_decision(tool_selection=...)` kwarg. Snapshot resets on
  successful enqueue and inside `reset()`.
- README link to the [`axor-core`](https://github.com/Bucha11/axor-core)
  repo from the intro and Requirements section.

### Changed
- `[telemetry]` extra pin bumped to `axor-telemetry>=0.3.0,<0.4` —
  the new `tool_selection` kwarg requires the matching wire-format
  field.

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
