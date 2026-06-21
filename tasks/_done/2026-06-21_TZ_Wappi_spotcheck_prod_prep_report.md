# Wappi spot-check и подготовка боевого долива

Дата: 2026-06-21

Ветка: `codex/wappi-history`

## Итог

Production apply не запускался. Боевая customer timeline БД не открывалась на запись.

Spot-check 2 привязанных Wappi-сессий подтверждён read-only:

| Сессия | Профиль | Канал | Бренд | Сообщений | Проверка |
|---:|---|---|---|---:|---|
| 1 | `ec2eed50-b55f` | Telegram | foton | 6 | один customer, один AMO contact, один активный lead, бренд lead=foton |
| 2 | `18b255b8-7a67` | Telegram | unpk | 49 | один customer, один AMO contact, один активный lead, бренд lead=unpk |

Для обеих сессий:

- AMO contact содержит тот же Telegram ID, по которому сделана привязка;
- AMO lead активный;
- contact входит в lead;
- бренд lead совпадает с брендом Wappi-профиля;
- `allowed_for_bot=0` у всех Wappi chunks/events;
- `match_key=Telegram ID`;
- полные chat/message/customer/lead/contact ID в отчёт не вынесены.

Примечание: у Foton-сессии `opportunity_id` в timeline-событии пустой, но `lead_id` сохранён в metadata, AMO read-only подтвердил активный lead и совпадение бренда. Это не блокер для текущего manager-only долива, но в production readback нужно отдельно проверить, что будущие потребители используют `metadata.lead_id`, если `opportunity_id` пустой.

## Cursor входа

Создан ignored cursor manifest:

`product_data/customer_timeline/canonical_readonly_wappi_resolver_final2_testcopy_20260621T120450Z/wappi_prod_apply_cursor_manifest_20260621T165900Z.json`

SHA256: `39cd84fa93830c9aff6931a759cbbc0b780653d45aa3157c14df89f3a8d8511c`

Содержит только безопасные хэши source id/source ref/chat/customer/lead, агрегаты по профилям и `input_hash` импортёра. Raw chat id, message id, телефоны, email и тексты сообщений не сохранялись.

Авторитетный входной курсор для production gate:

- `records_built=842`;
- `linked_by_amo_auto=55`;
- `pending_attribution=787`;
- `wappi_telegram` input hash: `9792737c16a8be410d0532fb651a5b643716cbfd5f23a77486994f74239cc0a9`;
- `wappi_max` input hash: `acfc84ab64247f1a9bb47ebc0ce6082e33a3682419fd3446b0ec339a7c7ea14d`.

Важно: текущая test-copy DB после repeat содержит 788 pending-conflict, потому что repeat увидел 1 новый pending из живого Wappi/AMO окна. Поэтому для боевого решения сравнивать нужно dry-run input hash/counts, а не молча принимать расхождение.

## Чек-лист боевого долива

Боевая БД по свежим D1-отчётам:

`/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_timeline/customer_timeline_prod_20260621/customer_timeline.sqlite`

До apply:

1. Получить отдельную отмашку Дмитрия на production apply.
2. Проверить ветку/commit Wappi-кода и чистый `git status`.
3. Запустить Wappi dry-run на production seed с теми же утверждёнными лимитами и AMO auto-resolver.
4. Сравнить dry-run с cursor manifest: input hashes, `records_built`, `linked_by_amo_auto`, `pending_attribution`, counts по 4 профилям.
5. Если Wappi live-source дрейфует и counts/hash отличаются, остановиться и отдать Дмитрию/Claude diff, без apply.
6. Создать SQLite backup через `.backup`/штатный backup helper в:
   `.../customer_timeline_prod_20260621/backups/before_wappi_ingest_<ts>/customer_timeline.sqlite`.
7. Проверить `PRAGMA quick_check` на source и backup.
8. Только после этого запускать apply.

После apply:

1. `PRAGMA quick_check = ok`.
2. Wappi chunks `allowed_for_bot=1`: 0.
3. Дубликаты Wappi events по `(source_system, source_id)`: 0.
4. Дубликаты Wappi chunks по `(source_system, source_ref)`: 0.
5. `blocked_customer_relink_conflicts=0`.
6. Readback counts совпадают с apply report.
7. AMO/Tallanto/CRM writes: 0; Wappi sent messages: 0.
8. Отчёт не содержит email/full phone/raw text.

Откат:

1. Не запускать новые writer-процессы на timeline до решения.
2. Сохранить failed apply report и `quick_check`.
3. Восстановить production DB из backup целиком или переключить путь на backup по утверждённой процедуре D1.
4. Повторить `PRAGMA quick_check`.
5. Повторить readback counts до Wappi apply и убедиться, что Wappi events/chunks/conflicts вернулись к состоянию backup.

## NEG

- Production apply: не запускался.
- Боевая DB: не менялась.
- Wappi send: 0.
- AMO/Tallanto/CRM write: 0.
- AMO spot-check: 4 read-only MCP `amo_api_get` вызова.
- Manifest git-ignored.
- Manifest PII grep: email 0, raw phone 0.
- Test-copy SQLite: `quick_check=ok`.
- Wappi chunks `allowed_for_bot=1`: 0.
- Wappi event duplicates: 0.
- Wappi chunk duplicates: 0.

## Проверки

- `python3 scripts/preflight.py --tz tasks/_running/2026-06-21_TZ_Wappi_spotcheck_prod_prep.md`: OK.
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_wappi_history_import_to_timeline.py tests/test_amo_wappi_auto_resolver.py tests/test_amo_wappi_transport.py tests/test_run_amo_wappi_draft_loop.py`: 24 passed.
- Semantic review: `PASS_WITH_NOTES`; блокеров для подготовки нет, production apply остаётся только по отдельной отмашке после свежего dry-run.

## Артефакты

- Cursor manifest: ignored path выше.
- Audit pack: `audits/_inbox/wappi_spotcheck_prod_prep_<ts>/`.
