# Claude Code Audit Context

This repository uses Claude Code as an independent read-only auditor and Codex as the primary implementation agent.

## Current Phase

Post-backfill CRM/writeback readiness after Stage15 v11 frozen gate.

## Current Step

Audit newly rebuilt CRM/AMO-ready input and verify it uses the post-backfill phone-chain layer, not the old April export pointer.

Before making a verdict, always check the newest formal artifacts in:

- `docs/`
- `audits/_inbox/<pack>/`
- `stable_runtime/` only when the audit pack explicitly references it

Do not rely on informal chat history.

## Seven-Point Safety Plan

Status on 2026-05-10:

1. Fixpoint sanitizer + idempotence test: done.
2. `docs/THREAT_MODEL.md` with leak classes: current canonical threat model.
3. Frozen adversarial corpus: done.
   - release corpus: `stable_runtime/bot_safety_frozen_corpus_20260510_v3_frozen_gate/bot_safety_adversarial_cases.jsonl`
   - validation: `stable_runtime/bot_safety_frozen_corpus_validation_20260510_v4_frozen_gate/summary.json`
4. ASR-tolerance layer: done in frozen corpus.
5. Separate sanitizer and detector: done.
   - detector: `src/mango_mvp/quality/bot_safety_detector.py`
   - detector must not import sanitizer regexes.
6. Heuristic NER layer: implemented as deterministic heuristic coverage; Natasha is intentionally not introduced.
7. Exit criterion for controlled manager-assist allowlist: met for Stage15 v11.
   - Stage15: `stable_runtime/transcript_quality_stage15_export_gate_20260510_v11_frozen_gate/summary.json`
   - frozen corpus validation: passed, 1312 rows, 0 failures.
   - autonomous bot remains blocked until over-sanitization queue is reviewed.

Current open risk is not Stage15 safety. Current open risk is that legacy CRM/writeback paths may still read `stable_runtime/CANONICAL_EXPORT.txt`, which previously pointed to an old April `sales_master_export_*` directory.

## Repository Map

- Implementation code: `src/mango_mvp/`
- Tests: `tests/`
- Scripts: `scripts/`
- Product and architecture docs: `docs/`
- Runtime artifacts: `stable_runtime/`
- Claude audit inbox packages: `audits/_inbox/`
- Claude audit results: `audits/_results/`
- Claude slash commands: `.claude/commands/`

## Authority Split

Codex owns implementation and test changes.

Claude Code owns only:

- `audits/_results/`

Claude Code must not edit:

- `src/`
- `tests/`
- `scripts/`
- `docs/`
- `stable_runtime/`
- `audits/_inbox/`
- `CLAUDE.md`
- `.claude/commands/`

Codex must not edit completed Claude result folders in `audits/_results/`.

Communication between agents must happen only through formal artifacts:

- threat model
- audit pack
- findings CSV
- row decisions CSV
- audit result markdown

Do not use shared free-form chat logs as source of truth.

## Safety Boundaries

During audits:

- Do not write to CRM/AMO/Tallanto.
- Do not run ASR.
- Do not run R+A processing.
- Do not delete files.
- Do not modify `stable_runtime/`.
- Do not expand audit scope indefinitely.
- If a new class of issue appears, classify it as either:
  - known threat-model class
  - new future threat-model class

## Last External Audit Context

Historical handoff context from the previous Claude/Cowork audit:

- Stage 15 v3 passed the control check.
- The architecture iteration was then implemented through Stage15 v11 frozen gate.
- Controlled manager-assist allowlist is considered quality-ready.
- Autonomous bot production is not ready until over-sanitization queue and company policies are reviewed.
- Before CRM writeback, verify that AMO-ready/contact context inputs are rebuilt from post-backfill artifacts:
  - `stable_runtime/canonical_master_20260510_after_quality_backfill_v1/canonical_calls_master.db`
  - `stable_runtime/insight_readiness_report_after_quality_backfill_20260510_v1/client_chains.csv`

If newer repository docs contradict this handoff, prefer the newest formal repository docs and the current audit pack.
