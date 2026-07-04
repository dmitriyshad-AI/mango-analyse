# Backward compatibility

## OFF behavior

`TELEGRAM_DIALOG_SUMMARY_ROLLING` default is OFF and is not added to the pilot profile.

When OFF:

- direct-path prompt does not include `dialog_summary`;
- `_normalize_direct_path_payload` ignores any incidental `dialog_summary`;
- `update_dialogue_memory_after_answer` does not write the rolling summary branch;
- Wappi history falls back to existing raw history summary behavior.

## API compatibility

`update_dialogue_memory_after_answer` gains optional keyword-only-style parameters at the end of the signature:

- `dialog_summary: str | None = None`
- `context: Mapping[str, Any] | None = None`

Existing callers remain valid. The simulator, draft loop, public Telegram pilot runtime, and live-check smoke were updated to pass the new value where available.

## Snapshot/moratorium

`tests/fixtures/adr003_direct_path_text_patterns_snapshot.json` was intentionally refreshed for two technical provider string checks:

- `"dialog_summary" in prompt_text`
- `"ПРЕДЫДУЩАЯ СВОДКА" in prompt_text`

These checks decide whether to normalize the additive JSON field; they do not classify client meaning.
