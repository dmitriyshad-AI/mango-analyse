# Readback plan

Для будущего live после отдельного "да":

1. Перед PATCH повторить этот dry-run на тех же `customer_id`.
2. Проверить, что `active_brand/open_deal_count` заполнены у всех lead rows.
3. Проверить `pre_patch_status=allowed`, `ready_for_write=yes`.
4. После PATCH сделать readback только целевых allowlist-полей:
   - contact: `AI-рекомендованный следующий шаг`, `Последняя AI-сводка`, `Авто история общения`;
   - lead: `AI-рекомендованный следующий шаг`, `AI-сводка по сделке`, `AI-история по сделке`.
5. Сверить sha old/new/actual.
6. При mismatch не повторять PATCH автоматически; отправить в manual review.
