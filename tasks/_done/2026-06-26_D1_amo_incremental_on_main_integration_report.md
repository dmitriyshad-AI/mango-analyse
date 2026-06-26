# D1 AMO incremental on main integration report

Дата: 2026-06-26  
Repo: `/Users/dmitrijfabarisov/Projects/Mango_main_intent_ff`  
Base before this block: `main@4f35626`  

## Что сделано

Точечно перенесён AMO incremental код из D3-ветки `codex/tz-c-nightly-cursors` без вливания всей старой ветки:

- `scripts/run_customer_timeline_amo_incremental.py`
- `src/mango_mvp/customer_timeline/amo_incremental.py`
- `tests/test_customer_timeline_amo_incremental.py`
- минимальные изменения:
  - `scripts/run_customer_timeline_nightly_incremental.py`
  - `src/mango_mvp/customer_timeline/ingestion.py`
  - `src/mango_mvp/customer_timeline/nightly_incremental.py`

Старая D3-ветка целиком не вливалась, потому что её полный diff относительно текущего `main` содержит удаления/откаты несвязанных Wappi/P0/bot-safe/registry файлов.

## Production timeline read-only facts

Боевая SQLite, только чтение:

`/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_timeline/customer_timeline_prod_20260621/customer_timeline.sqlite`

- SHA256: `ef9ef249b4192b768cd1eb826f6df20514994539a3911f9aeee19bbc295d03c8`
- Size: `2703142912`
- WAL: `0` bytes
- `PRAGMA integrity_check`: `ok`
- `timeline_events`: `158715`
- `bot_context_chunks`: `126983`
- `identity_links`: `84924`
- `ingestion_cursors`: `0`
- `ingestion_runs`: `7`

Source counts:

- `amocrm_snapshot`: `16277`, latest `2026-05-13T11:17:49+00:00`
- `mail_archive`: `4168`
- `mail_archive_stage2`: `30093`
- `mango_processed_summary`: `72998`, latest `2026-06-25T13:35:43+00:00`
- `master_contacts_snapshot`: `16901`
- `tallanto_snapshot`: `16901`
- `telegram_history`: `1377`

Вывод: AMO-часть истории действительно устарела; курсоров ещё нет.

## Test-copy real smoke

Запуск был только на копии БД, AMO read-only GET, без production write:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/run_customer_timeline_amo_incremental.py \
  --source-db "/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_timeline/customer_timeline_prod_20260621/customer_timeline.sqlite" \
  --out-root "/Users/dmitrijfabarisov/.mango_local/amo_incremental_gate_20260626_134725" \
  --page-limit 5 \
  --max-pages 1 \
  --since "2026-06-25T00:00:00+03:00" \
  --summary-only
```

Артефакты:

- Test copy: `/Users/dmitrijfabarisov/.mango_local/amo_incremental_gate_20260626_134725/customer_timeline.sqlite`
- Full report: `/Users/dmitrijfabarisov/.mango_local/amo_incremental_gate_20260626_134725/amo_incremental_report.json`
- Summary stdout: `/Users/dmitrijfabarisov/.mango_local/amo_incremental_gate_20260626_134725/summary_stdout.json`

Smoke results:

- leads fetched/normalized: `5 / 5`
- contacts fetched/normalized: `5 / 5`
- events fetched/normalized: `5 / 0`
- first run changed_customer_count: `10`
- repeat run changed_customer_count: `0`
- safety:
  - AMO write: `false`
  - Tallanto write: `false`
  - CRM write: `false`
  - notes endpoint used: `false`
  - bot_safe_summary_created: `false`
  - test_copy_only: `true`

Почему events normalized = 0 в этом smoke: маленький `--max-pages 1 --page-limit 5` срез `/events` попал на сущности, которых не было в текущем одностраничном `leads/contacts updated_at` срезе; все 5 событий корректно ушли в `unmatched`, а не были привязаны наугад.

## Тесты

Целевые:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q \
  tests/test_customer_timeline_amo_incremental.py \
  tests/test_customer_timeline_nightly_incremental.py \
  tests/test_customer_timeline_ingestion.py

25 passed
```

Полный pytest:

```text
3657 passed, 5 skipped, 1 warning in 81.13s
```

## Границы

- Production `customer_timeline.sqlite`: не писалась
- AMO/Tallanto/CRM write: `0`
- AMO network: только read-only GET в test-copy smoke
- `stable_runtime`: не трогался
- `git reset/checkout/clean`: не использовались

## Следующий gate

Production apply всё ещё НЕ выполнен.

Перед ним нужен отдельный свежий controlled-apply план/разрешение:

1. backup production SQLite с SHA-манифестом;
2. полный test-copy прогон с достаточным `page_limit/max_pages`;
3. проверка event→customer по всему apply-набору;
4. повторный прогон = `changed_customer_count=0`;
5. только после отдельного “да” Дмитрия — production apply.
