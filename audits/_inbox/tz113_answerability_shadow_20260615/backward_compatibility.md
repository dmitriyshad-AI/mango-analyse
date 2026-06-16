# Backward compatibility

- `TELEGRAM_ANSWERABILITY_SHADOW` выключен по умолчанию.
- `pilot_gold_v1` не включает новый флаг.
- Старые вызовы `_normalize_direct_path_payload(payload, raw_response=...)` совместимы: новый параметр `include_answerability_self` имеет значение `False`.
- Существующие наследники `SubscriptionLlmDraftProvider`, которые переопределяют `_direct_path_draft_runner(prompt)`, не требуют изменения сигнатуры.
- `dynamic_summary.json` получает новый раздел `answerability_trace` только при наличии следа.
