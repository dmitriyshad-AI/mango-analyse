# Обратная совместимость

- `CRM_CARD_AGGREGATOR_ENABLED=OFF`: contact writer и deal payload остаются на старом поведении.
- `CRM_AUTO_HISTORY_CHRONOLOGY_TEXT=OFF`: `_compose_auto_history` оставляет прежний marker-блок хронологии.
- `CRM_DEAL_OBJECTION_EXPLICIT_COMPACT=OFF`: `compact_objection` использует старый ellipsis.
- `CRM_CONTACT_WRITEBACK_AI_ALLOWLIST=OFF`: нижний AMO helper сохраняет прежний protected-only guard.
- Новые модули `crm_card_aggregator` и `crm_card_workbook` не меняют `customer_timeline` и не создают runtime-хранилищ.
- Workbook/CSV/summary являются transient artifacts и не коммитятся.
