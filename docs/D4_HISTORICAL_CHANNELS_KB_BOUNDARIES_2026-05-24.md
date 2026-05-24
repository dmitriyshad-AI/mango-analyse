# D4 Historical Channels KB Boundaries

Date: 2026-05-24

Task source: `Claude Cowork Space/02_dialogue_tasks/D4_historical_channels_kb_TZ.md`.

## Working Tree State

The repository already has unrelated changes before D4 work:

- mass deletions of old KB releases and old audit packs;
- modified docs, runtime and audio scripts from parallel work;
- modified v6.3 KB generated artifacts and source YAML;
- untracked `Claude Cowork Space/`, `product_data/bot_improvement_candidates_20260523/`, and `src/mango_mvp/channels/p0_recall_spec.py`.

These changes are treated as external to D4 unless explicitly listed below.

## D4 Allowlist

D4 may change only these areas:

- `docs/D4_HISTORICAL_CHANNELS_KB_BOUNDARIES_2026-05-24.md`
- `docs/kb_candidate_approval_queue_2026-05-24/`
- `scripts/build_kb_release_v6_1_team_answers.py`
- `scripts/build_kb_release_v3_from_claude_handoff.py`
- `product_data/knowledge_base/kb_release_20260520_v6_3_team_answers_sources/release_manifest.yaml`
- `scripts/run_telegram_stage6_kb_eval.py`
- `product_data/knowledge_base/kb_release_20260520_v6_3_team_answers_sources/facts/facts_for_bot_FOTON.yaml`
- KB-only tests under `tests/test_kb_*.py`
- D4 audit pack under `audits/_inbox/d4_historical_channels_kb_20260524_*/`
- Stage6/smoke audit pack under `audits/_inbox/kb_v63_stage6_smoke50_*/`

## Explicitly Out Of Scope

- `channels/` and `src/mango_mvp/channels/`
- `stable_runtime/`
- old release deletions in `product_data/knowledge_base/kb_release_20260517_*` and `kb_release_20260518_*`
- AMO/CRM/Tallanto writes
- Telegram live sends, polling restarts, or write operations
- ASR and Resolve+Analyze
- raw email, raw Telegram exports, raw call transcripts, attachments, and personal data

## P0 Recall Contract

Do not create a new D4 critical-topic list. D4 references the canonical artifact from the main dialogue:

- `src/mango_mvp/channels/p0_recall_spec.py`

The file is currently untracked and belongs to the main dialogue. D4 may reference it in docs or tests, but must not edit or stage it.

## Historical Channel Enrichment Rule

Email, Telegram, calls and attachments may only create masked approval candidates:

`candidate -> status -> human approval -> later KB task`

No historical item may auto-add a KB fact. Manager answers are style/evidence only, not facts. Each candidate must be single-brand and must not contain personal data or verbatim manager replies.
