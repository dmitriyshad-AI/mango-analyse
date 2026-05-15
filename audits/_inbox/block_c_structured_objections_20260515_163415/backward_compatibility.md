# Backward Compatibility

## Сохраняется

- `AI-актуальные возражения` остается строкой.
- Top-8 для менеджерского поля сохраняется.
- При отсутствии возражений сохраняется старый fallback:
  `Актуальные возражения в релевантных звонках не выделены.`
- AMO payload не получает `AI-возражения структура`.

## Добавлено

- `structured_objections_json` в preview row;
- `structured_objections_count`;
- `objections_truncated`;
- `structured_objections` в payloads JSONL.
