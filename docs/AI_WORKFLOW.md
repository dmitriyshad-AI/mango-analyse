# AI Workflow

Дата: 2026-05-10

## Purpose

Этот документ фиксирует рабочий процесс, где Codex остается основным разработчиком, а Claude Code CLI используется как независимый аудитор в той же репе.

Цель: убрать ручную упаковку audit-пакетов для Cowork и оставить при этом независимость аудита.

## Roles

Codex owns:

- `src/`
- `tests/`
- `scripts/`
- `stable_runtime/`
- `docs/THREAT_MODEL.md`
- `docs/AI_WORKFLOW.md`
- `CLAUDE.md`
- `.claude/commands/`
- `audits/_inbox/`

Claude Code owns:

- `audits/_results/`

Codex does not edit completed Claude result folders in `audits/_results/`.

Claude Code does not edit anything outside `audits/_results/`.

This initial scaffold is the only exception: Codex creates the empty `audits/_results/` directory and README workflow.

## Communication Contract

Agents communicate only through formal artifacts:

- `docs/THREAT_MODEL.md`
- current Stage15 / frozen-corpus / fixpoint summaries referenced by the pack
- audit packs in `audits/_inbox/`
- `CLAUDE_REAUDIT_RESULT.md`
- `findings.csv`
- `row_decisions.csv`

Do not use free-form shared chat logs as source of truth.

## Iteration Cycle

1. Codex implements a focused change.
2. Codex runs tests and safety checks.
3. Codex commits the implementation.
4. Codex writes a new audit pack to `audits/_inbox/<phase_name>/`.
5. Dmitry runs Claude Code from repo root:

   ```bash
   claude /audit audits/_inbox/<phase_name>
   ```

6. Claude Code reads `CLAUDE.md`, `.claude/commands/audit.md`, `docs/THREAT_MODEL.md` and the audit pack.
7. Claude Code writes results to `audits/_results/<YYYY-MM-DD>_<phase_name>/`.
8. Claude Code updates only `audits/_results/<YYYY-MM-DD>_<phase_name>/`.
9. Codex reads results and decides which fixes to implement.
10. Repeat.

## Required Claude Result Files

Each audit result folder should contain:

- `CLAUDE_REAUDIT_RESULT.md`
- `findings.csv`
- `row_decisions.csv`

Optional files:

- `notes.md`
- `commands_run.txt`
- `sampled_rows.csv`

## Last Audit Pointer

`audits/README.md` should contain one human-readable line:

```text
Last audit: <date> <phase> <verdict>
```

Claude Code must not update that line because it is outside `audits/_results/`.
After reading a completed result folder, Codex may update the line in a separate implementation pass.

## Hard Safety Rules

Claude Code must not:

- write to CRM/AMO/Tallanto;
- run ASR;
- run R+A;
- delete files;
- mutate `stable_runtime/`;
- edit implementation files;
- edit tests;

## Current Release References

As of 2026-05-10, the current transcript-quality release references are:

- Stage15: `stable_runtime/transcript_quality_stage15_export_gate_20260510_v11_frozen_gate/summary.json`
- Frozen corpus: `stable_runtime/bot_safety_frozen_corpus_20260510_v3_frozen_gate/bot_safety_adversarial_cases.jsonl`
- Frozen validation: `stable_runtime/bot_safety_frozen_corpus_validation_20260510_v4_frozen_gate/summary.json`
- Canonical post-backfill DB: `stable_runtime/canonical_master_20260510_after_quality_backfill_v1/canonical_calls_master.db`

CRM/writeback audits must verify that new inputs are derived from this post-backfill layer and not from the old April `sales_master_export_*` pointer.
- edit docs outside `audits/_results/`.

Codex must not:

- rewrite Claude findings;
- silently delete audit results;
- use informal chat claims as replacement for formal findings.

## Version Check

Before the first run:

```bash
claude --version
```

Recommended minimum:

```text
2.1.126
```
