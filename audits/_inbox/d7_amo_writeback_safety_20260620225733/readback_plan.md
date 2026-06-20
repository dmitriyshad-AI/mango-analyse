# Readback plan

Live readback не выполнялся: текущий блок только DRY-RUN и unit/fake-контур.

План для первого разрешенного live-этапа:

1. Перед live PATCH проверить, что входной CSV, approval и snapshot manifest совпадают по sha256.
2. После live PATCH читать те же AMO сущности через GET:
   - lead rows: `fetch_lead(lead_id)`;
   - contact rows: `fetch_contact(contact_id)`.
3. Сверить только allowlist-поля:
   - deal: `LEAD_WRITE_ALLOWED_FIELDS`;
   - contact: `CONTACT_WRITE_ALLOWED_FIELDS`.
4. Для каждого поля записать `expected_sha`, `actual_sha`, `matched`, `entity_type`, `entity_id`, `snapshot_key`.
5. Если `matched=false`, не делать автоматический повторный PATCH; строку отправить в manual review.
6. Если после PATCH менеджер уже изменил поле, rollback должен дать `current_value_changed_after_write`, а не перетереть ручную правку.
7. Protected-поля (`Телефон`, `ФИО`, `Email`, ручная `История общения`, статус/этап/воронка/ответственный, Tallanto-поля) в readback только проверяются как "не менялись"; они не входят в payload.

Критерий успешного readback: все реально записанные allowlist-поля совпали по sha, protected-поля не попали в PATCH payload, расхождения ушли в ручной разбор.
