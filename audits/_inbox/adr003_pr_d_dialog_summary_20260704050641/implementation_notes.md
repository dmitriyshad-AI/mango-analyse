# PR-D rolling dialog summary — implementation notes

## Scope

Implemented PR-D from `Foton/2026-07-04_TZ_PR-D_pamyat_dialoga_rolling_svodka_dlya_D1.md`.

One new default-OFF flag:

- `TELEGRAM_DIALOG_SUMMARY_ROLLING`

The flag is not added to the pilot profile. It must be enabled explicitly for measurements.

## What changed

- `build_dialogue_memory(...)` preserves existing `conversation_summary_short` when the rolling flag is ON and previous memory has a safe summary. OFF keeps the old slot-glue fallback.
- Direct-path prompt gets an additive `dialog_summary` JSON field only when the rolling flag is ON. It is produced by the same direct-path LLM call; no extra model call was added.
- Direct-path payload normalization stores `dialog_summary_candidate` in metadata only when the prompt actually requested the field.
- `update_dialogue_memory_after_answer(..., dialog_summary=..., context=...)` writes a safe candidate into `conversation_summary_short` before the `memory_provenance` early return. Empty/unsafe candidates are ignored.
- Wappi prompt history uses persisted rolling summary first when ON, with raw history summary as fallback.
- Dynamic simulator post-answer update paths now pass `dialog_summary` into memory updates.
- Public Telegram pilot and live-check post-answer update paths now pass `dialog_summary` into memory updates.
- Added a draft long-persona set for PR-D directional measurements; it is not a canon set.

## Safety filters reused

- Summary text cleanup and unsafe facts: existing `_memory_llm_summary(...)` / `_MEMORY_LLM_UNSAFE_SUMMARY_FACT_RE`.
- PII: existing direct-path support phone/email regexes imported late to avoid import cycles.
- Foreign brand: local brand-token guard without new `re.compile`.
- Unsupported bare numbers/dates/percent words: deterministic fail-closed guard rejects model summaries with long digit groups, numeric dates, and `процент`.

No new `re.compile` was added in `dialogue_memory.py`.

## Local directional checks

- Targeted PR-D/public/live-check tests after auditor fixes: `740 passed`.
- Full pytest after auditor fixes: `4052 passed, 5 skipped, 1 warning` (final run: 82.15s).
- Fake schema/runner smoke on all 6 draft long personas: `6 dialogs`, `60 turns`, `hard_gate_failures=0`.
- Direct-path ON one-turn smoke: provider error empty; `direct_path.dialog_summary_candidate` present; `conversation_summary_short` updated from the candidate.
- Direct-path OFF one-turn smoke: provider error empty; `dialog_summary_candidate=None`; `conversation_summary_short` empty.

## Known limitation

The local real bot pair was reduced to a one-turn direct-path smoke. Full semantic acceptance on long personas is intentionally deferred to Fable/Claude source regreyde in the morning.

## Auditor findings addressed

Independent auditor initially returned `BLOCKED` for two reasons:

1. Public Telegram pilot and `check_public_bot_live.py` did not pass `dialog_summary`/`context` into `update_dialogue_memory_after_answer`.
2. Numeric fail-closed filtering accepted bare amounts, numeric dates, and percent words.

Both were fixed before commit and covered by tests.
