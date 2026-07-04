# Local measurements

## Fake schema/runner smoke

Command class: scripted client, fake judge, fake bot/memory/semantic modes.

Output:

- `/tmp/adr003_prd_dialog_summary_fake_smoke_20260704_045937`

Result:

- dialogs: 6
- turns: 60
- fail: 0
- hard_gate_failures: 0
- llm_calls.total: 0

Meaning: draft long-persona set is syntactically runnable and does not break the simulator.

## Inconclusive real pair attempt

Output:

- `/tmp/adr003_prd_dialog_summary_real_smoke_20260704_050049`

Result:

- OFF and ON completed, but direct-path metadata was empty and several turns had `codex_retryable_error` / `llm_fallback`.

Meaning: this run is not valid for judging PR-D. It is kept only as infra evidence; it did not drive code decisions.

## Direct-path ON one-turn smoke

Env:

- `TELEGRAM_DIRECT_PATH=1`
- `TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1`
- `TELEGRAM_DIALOG_SUMMARY_ROLLING=1`

Output:

- `/tmp/adr003_prd_dialog_summary_direct_on_20260704_050319`

Observed:

- provider_error: empty
- route: `bot_answer_self_for_pilot`
- `direct_path_model`: present in safety flags
- `direct_path.dialog_summary_candidate`: present
- `bot_dialogue_memory_after_answer.conversation_summary_short`: updated to the same safe summary

Candidate:

```
Родитель ищет для ребёнка онлайн-физику в Фотоне. Ребёнок перешёл в 8 класс; обсуждается регулярный онлайн-курс по физике.
```

## Direct-path OFF one-turn smoke

Env:

- `TELEGRAM_DIRECT_PATH=1`
- `TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1`
- no `TELEGRAM_DIALOG_SUMMARY_ROLLING`

Output:

- `/tmp/adr003_prd_dialog_summary_direct_off_20260704_050405`

Observed:

- provider_error: empty
- route: `bot_answer_self_for_pilot`
- `direct_path_model`: present in safety flags
- `direct_path.dialog_summary_candidate`: `None`
- `bot_dialogue_memory_after_answer.conversation_summary_short`: empty

Meaning: default-OFF behavior is preserved for the new field.
