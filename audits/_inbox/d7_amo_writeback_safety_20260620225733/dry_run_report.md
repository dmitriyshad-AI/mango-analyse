# Dry-run

Live AMO write не запускался. Dry-run демонстрация выполнена через fake-тестовый контур.

## Anti-clobber demo

Тест: `tests/test_deal_aware_amo_rollback.py::test_live_write_blocks_patch_when_current_value_changed_after_snapshot`.

Сценарий:

1. Первый fake GET возвращает старые значения и строится snapshot.
2. Второй fresh fake GET перед PATCH возвращает `manual ...`.
3. Safety layer сравнивает sha и возвращает `clobber_protected`.
4. `send_update_func` не вызывается.

Проверка:

```text
report["rows"][0]["status"] == "skipped"
"clobber_protected" in report["rows"][0]["reason"]
"patch" not in events
```

## Snapshot / journal demo

Тесты:

- `tests/test_amo_write_safety.py::test_contact_snapshot_rows_keep_entity_identity`
- `tests/test_amo_write_safety.py::test_dry_run_journal_does_not_become_last_written_sha`

Смысл:

- contact rows имеют `entity_type=contact`, `entity_id=<contact_id>`, но сохраняются в том же `pre_write_snapshot.jsonl/csv`;
- `written-dry` пишется в journal как аудит, но не используется как last-written sha для будущего anti-clobber;
- только `action=written` считается нашим последним фактическим значением.

## Contact rollback demo

Тест: `tests/test_deal_aware_amo_rollback.py::test_rollback_apply_can_restore_contact_snapshot`.

Сценарий:

- snapshot row: `entity_type=contact`, поле `Авто история общения`;
- fake `fetch_entity(contact, 777)` возвращает текущее `new`;
- fake `send_entity_update(contact, 777, old)` вызывается;
- live AMO не используется.

Проверка:

```text
rollback_status == "restored"
calls == [{"entity_type": "contact", "entity_id": 777, "field_payload": {"Авто история общения": "old"}}]
```
