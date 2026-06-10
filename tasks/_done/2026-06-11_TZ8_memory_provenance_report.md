# TZ-8 memory provenance report

Date: 2026-06-11
Branch: `codex/preserve-wave6-profile-final-dirty`

## Scope

Implemented `tasks/_inbox_codex/2026-06-10_TZ8_memory_provenance_DRAFT.md` behind default-OFF flag:

```text
TELEGRAM_MEMORY_PROVENANCE
```

The pilot profile was not changed.

## Read-only map

- Memory model path:
  - `src/mango_mvp/channels/dialogue_memory.py`
  - `build_dialogue_memory()`
  - `update_dialogue_memory_after_answer()`
  - `update_memory_llm()` / `build_memory_llm_prompt()`
- Dynamic runner:
  - `scripts/run_telegram_dynamic_client_sim.py`
  - `build_memory_model()` creates the counted memory model.
- Context assembly:
  - `src/mango_mvp/channels/telegram_pilot_context_builder.py`
  - `src/mango_mvp/pilot_context_assembly.py`
  - `src/mango_mvp/channels/pilot_context.py`
- Live draft loop:
  - `src/mango_mvp/integrations/draft_loop.py`
  - `scripts/run_amo_wappi_draft_loop.py`

## Implementation

- Added deterministic provenance fields to memory slots:
  - `quote`
  - `turn_index`
  - `message_id`
  - `child_key`
- Added slot history for superseded values.
- Added v1 extractor for:
  - `grade`
  - `subject`
  - `format`
  - `location`
  - `child_name`
  - `payment_pref`
  - simple two-child grade split via `child_1_grade` / `child_2_grade`
- With `TELEGRAM_MEMORY_PROVENANCE=1`, memory slots are written only from client turns and only rendered into prompt when they have a quote.
- With `TELEGRAM_MEMORY_PROVENANCE=1`, memory LLM is not constructed in the dynamic runner, so `llm_calls.memory` should be 0.
- Added full technical `dialogue_memory_state` alongside compact `dialogue_memory_view`.
- Draft loop state now persists `dialogue_memory` by `(profile_id, chat_id)` and carries `message_id` from Wappi messages.

## NEG coverage

- Slot without quote is not rendered into prompt.
- Bot/manager-style history lines do not create client slots.
- Conflict value supersedes the old slot and keeps old slot in history.
- Two children in one client turn keep separate child grade slots.
- Provenance mode skips memory LLM even if a memory function is provided.
- Draft loop persists provenance memory under the composite Wappi key.
- Dynamic runner returns no memory model when provenance mode is enabled.

## Tests

Targeted:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_dialogue_memory.py tests/test_draft_loop.py tests/test_telegram_dynamic_client_sim.py -k 'memory_provenance or build_memory_model or draft_loop_persists'
8 passed, 138 deselected
```

Expanded affected files:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_dialogue_memory.py tests/test_draft_loop.py tests/test_run_amo_wappi_draft_loop.py tests/test_telegram_pilot_context_builder.py tests/test_pilot_context.py tests/test_telegram_dynamic_client_sim.py
188 passed
```

Full:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests
2927 passed, 5 skipped, 1 warning
```

## Residual notes

This is a formal pass. The flag is default OFF and not included in `pilot_gold_v1`.

Semantic/regression measurement is still required later on the paired 89 run described in the TZ.
