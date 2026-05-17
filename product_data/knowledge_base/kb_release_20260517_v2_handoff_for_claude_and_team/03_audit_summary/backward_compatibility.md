# Backward compatibility

- Старые v1 snapshots без brand-поля продолжают работать: legacy-факты не отбрасываются только из-за отсутствия `active_brand`.
- Для новых v2 snapshots брендовые факты фильтруются строго по `active_brand`.
- `draft_for_manager`, `manager_only`, `blocked` сохранены.
- Добавлен новый допустимый маршрут `bot_answer_self`, но текущий пилот всё равно остаётся в режиме черновиков и проверки менеджером.
- `fresh` и `fresh_verified` сохранены; добавлен `document_verified` для документально подтверждённых фактов.
- Stage 6 остаётся read-only dry-run: внешних write-операций нет.
