# Backward Compatibility

- Существующий формат `facts_registry.jsonl`, `quality_report.json`, `semantic_review.json`, `brand_rules.yaml`, `bot_policy.yaml` сохранён.
- Добавлены поля `manager_display_text` и `bot_template_registry.json`; старые потребители могут их игнорировать.
- `manager_check_text` сохранён для диагностики, но для показа менеджеру теперь нужно использовать `manager_display_text`.
- `bot_template_required` стал строже для цен и скидок. Это может чаще вести в шаблон или `manager_only`, но не ослабляет безопасность.
- Bot pack стал компактнее: полный snapshot и approval queue не входят в runtime-пакет.

