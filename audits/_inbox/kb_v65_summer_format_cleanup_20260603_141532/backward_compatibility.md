# Backward compatibility

- Старые `objection_responses.*` fact_key сохранены: cleanup сделан в renderer, а не сменой YAML-формы на `.client_safe_text`.
- Schedule-source v6.4 использован как база, поэтому required YAML paths `schedule_2026_27.groups` сохранены.
- Snapshot path в коде не менялся.
- Новые v6.5 факты добавлены отдельным namespace `kb_v6_5_client_safe_facts_2026_06_03.*`, не перетирая существующие ключи.
- Синхронизация старых ЛШ-структурных значений затронула только противоречащие v6.5 данные: УНПК 37 500 -> 39 500, Пацаева -> предзапись, active/former student discount -> 5%.

