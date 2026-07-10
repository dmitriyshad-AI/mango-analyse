# Что сделано

Ночная задача Н1-Н8 выполнена в режиме staging/local/read-only.

- Н1: пересобран transfer package на свежий CRM export `finish_n8_crm_export_after_nightly`.
- Н2: вручную отрепетирован nightly-service на staging, выполнен ограниченный AMO incremental read-only, пересчитаны signals/objections, собраны wave0 и full84 dossier.
- Н3: собраны книги досье по менеджерам через AMO read-only lookup.
- Н4: собран Wappi manual pairing workbook для 145 pending чатов; автопривязки нет.
- Н5: SWAP отрепетирован только на локальной копии, prod DB не подменялась.
- Н6: сделан AMO read-only diff для 3 ready карточек; записей нет.
- Н7: добавлена golden-negative pytest fixture для CRM writeback gates.
- Н8: пересобран свежий CRM export/transfer, выполнен smoke по 3 ready клиентам, полный pytest зелёный.

# Как проверялось

- Focused pytest: `79 passed, 1 warning`.
- Full pytest: `4036 passed, 5 skipped, 2 warnings`.
- Staging DB: `quick_check=ok`, `foreign_key_check_rows=0`.
- CRM export: `candidate_rows=66`, `ready_rows=3`, `blocked_rows=63`, idempotence passed.
- Transfer package sha совпадает со свежей staging DB: `887b3ad74b4943a1eea2abafc5ea627939858520638a7fb17f6f64d6beb1f7f8`.
- Prod DB sha до/после SWAP rehearsal не изменился: `ef9ef249b4192b768cd1eb826f6df20514994539a3911f9aeee19bbc295d03c8`.

# Что осталось

- Реальный SWAP не выполнять без отдельного окна и owner approval.
- AMO write не выполнять без anti-clobber/pre-patch gate и отдельного разрешения.
- Wappi workbook требует ручной разметки; `medium_text` не является точной привязкой.
- AMO incremental был ограничен `max_pages=2`; полноценный nightly требует отдельного решения.
