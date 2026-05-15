# Backward Compatibility

- Старый вызов `build_deal_text_preview(paths)` работает без изменений.
- Новые Stage 4 CLI-флаги необязательные.
- Без `--enable-customer-timeline-context` preview не получает новые timeline-поля.
- Stage 6/live writeback не требует customer timeline.
- `DEAL_AI_FIELDS`, обязательные AMO-поля и optional commercial fields не изменены.
- Старые deal-aware CSV/snapshot пути остаются рабочими.
