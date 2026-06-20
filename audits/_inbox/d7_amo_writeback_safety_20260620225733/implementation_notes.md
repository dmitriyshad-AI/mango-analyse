# Что сделано

- Добавлен общий safety-слой `src/mango_mvp/deal_aware/amo_write_safety.py`.
  - Перед PATCH сравнивает fresh GET из AMO с sha из snapshot/последнего реального journal write.
  - Возвращает `allowed`, `skipped:unchanged`, `clobber_protected`, `no_snapshot`.
  - Dry-run journal (`written-dry`) не считается нашим последним записанным значением.
- Расширен существующий `deal_aware.amo_rollback.build_pre_write_snapshot_rows`.
  - Сохранил прежний `lead_id` контракт.
  - Добавил `entity_type/entity_id`, чтобы тем же snapshot/rollback форматом покрывать contacts.
- Deal writer `scripts/write_deal_aware_amo_fields.py`.
  - После pre-write snapshot делает fresh GET и anti-clobber до PATCH.
  - PATCH получает только поля со статусом `allowed`.
  - При ручной правке/unchanged/no_snapshot пишет skipped и не вызывает PATCH.
  - Добавлен journal path.
- Contact writer `scripts/write_amo_ready_contacts.py`.
  - Contact payload сужен до 3 целевых полей ТЗ: `AI-рекомендованный следующий шаг`, `Последняя AI-сводка`, `Авто история общения`.
  - В dry-run/contact path добавлены `fetch_contact`, snapshot, fresh GET anti-clobber, journal.
  - Contact snapshots пишутся в тот же `pre_write_snapshot.jsonl/csv`.
- Rollback CLI `scripts/rollback_deal_aware_amo_fields.py`.
  - Добавлены `--contact`, `--lead`, `--snapshot`.
  - `entity_type=contact` rows идут через `fetch_contact/send_contact_custom_field_update`.
- Нижний AMO helper `src/mango_mvp/amocrm_runtime/amo_integration.py`.
  - Явный allowlist для contact/deal AI-полей.
  - Protected расширен: телефон, ФИО, email, ручная `История общения`, статус/этап/воронка/ответственный.
- Deal preflight `src/mango_mvp/deal_aware/deal_writeback.py`.
  - Блокирует `brand_conflict_channel_deal`.
  - Блокирует `multiple_open_deals`, если входная строка несёт такой счётчик.

# Что не делал

- Не запускал live AMO/Tallanto/CRM write.
- Не менял YAML.
- Не запускал ASR/Analyze/тяжёлые batch.
- Не писал в `stable_runtime` как рабочее состояние.

# Ключевые NEG

- Ручная правка после snapshot -> `clobber_protected`, PATCH не вызывается.
- Повтор без изменений -> `skipped:unchanged`, PATCH не вызывается.
- Contact snapshot откатывается тем же rollback runner через `entity_type=contact`.
- `Email`/`ФИО`/ручная `История общения`/status/pipeline/responsible не проходят нижний helper.
- Бренд канала не совпал с брендом сделки -> строка Stage6 blocked.
- Несколько открытых сделок -> строка Stage6 blocked.

# Статус

formal_pass: да.
semantic_pass: PASS_WITH_NOTES, потому что live не запускался и это правильно для текущего ТЗ.
