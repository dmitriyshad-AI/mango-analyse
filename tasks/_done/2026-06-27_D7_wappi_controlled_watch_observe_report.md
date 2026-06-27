# D7 Wappi controlled watch — шаг 0 observe-only

Дата: 2026-06-27  
Ветка/worktree: `/Users/dmitrijfabarisov/Projects/Mango_wappi_controlled_watch_observe`, `codex/wappi-controlled-watch-observe`  
Local main: fast-forward до `7ab8c43`  
Режим: observe-only, AMO/Tallanto/CRM write = 0, клиентские отправки = 0, AMO notes = 0.

## 1. Проверка плана и main

План: `/Users/dmitrijfabarisov/Claude Projects/Foton/2026-06-27_PLAN_Wappi_controlled_watch_osvezhyon.md`.

Проверено:

- В грязной главной папке `/Users/dmitrijfabarisov/Projects/Mango analyse` работать нельзя: она не на `main` и с большим числом чужих изменений.
- `cc0d9e6` как commit не входит в `main`, но его Wappi-delta уже была перенесена в `main` эквивалентным коммитом `386f08b Add Wappi draft-loop stabilization ops`.
- В `main` уже были: `scripts/wappi_draft_loop_ops.py`, `tests/test_wappi_draft_loop_ops.py`, `tests/test_wappi_stabilization_smoke.py`, endpoint-only test.
- Дополнительно найден стопор для постоянного observe-loop: `dry_run` не ставил watermark для обработанных dry-run сообщений/skip-ов. Это могло повторять один и тот же входящий поток в каждом цикле.

Исправлено и добавлено в `main`:

- `c6765f6 Deduplicate Wappi observe dry runs`
- `7ab8c43 Harden Wappi observe loop reporting`

## 2. Что изменено

Файлы:

- `src/mango_mvp/integrations/draft_loop.py`
  - В `dry_run` после `draft_created` сообщения помечаются processed локально.
  - В `dry_run` для `pair_missing` и `brand_pair_mismatch` тоже ставится локальный processed-watermark, чтобы observe-служба не повторяла один и тот же skip бесконечно.
  - Non-dry-run/live-write поведение не менялось.
- `scripts/wappi_draft_loop_ops.py`
  - Паспорт теперь предпочитает реальный Python runner, а не `screen` wrapper, если оба процесса содержат `run_amo_wappi_draft_loop.py` в командной строке.
  - `daily-report` подтягивает `auto_resolver_counts` из heartbeat, включая `not_enabled`.
- `tests/test_draft_loop.py`
  - NEG: повторный `dry_run` не дублирует `draft_created`.
  - NEG: повторный `dry_run` не дублирует `pair_missing`.
  - NEG: повторный `dry_run` не дублирует `brand_pair_mismatch`.
- `tests/test_wappi_draft_loop_ops.py`
  - NEG: паспорт выбирает Python runner, а не `screen`.
  - NEG: daily-report показывает auto-resolver причины из heartbeat.

## 3. Тесты

Targeted Wappi/D7:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q \
  tests/test_draft_loop.py \
  tests/test_run_amo_wappi_draft_loop.py \
  tests/test_wappi_draft_loop_ops.py \
  tests/test_wappi_stabilization_smoke.py \
  tests/test_amo_wappi_phase1.py \
  tests/test_conversation_intent_plan.py \
  tests/test_codex_exec_service_tier.py
```

Результат: `126 passed in 1.26s`.

Полный pytest:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q
```

Результат: `3692 passed, 5 skipped, 1 warning in 107.94s`.

Warning: `urllib3` на системном Python предупреждает про LibreSSL. На Wappi observe-логику это не влияет.

## 4. Запуск observe-службы

Служба запущена в `screen`:

- screen: `37740.mango_wappi_observe_20260627`
- runner PID: `37750`
- commit: `7ab8c43`
- команда runner: `python3 scripts/run_amo_wappi_draft_loop.py --loop --dry-run --interval-sec 45 --manager-outgoing-visible unknown`

Реальный env процесса по паспорту:

```text
DRAFT_LOOP_AUTO_RESOLVER=0
DRAFT_LOOP_AUTO_RESOLVER_ALLOW_ALL=0
TELEGRAM_BOT_SAFE_CRM_CONTEXT=0
TELEGRAM_MEMORY_PROVENANCE=0
```

`--live-write` не используется. Клиентских send-path в Wappi loop нет. AMO note write path не вызывался.

Аварийный стоп:

```bash
touch ~/.mango_secrets/STOP_DRAFT_LOOP
```

Текущее значение в паспорте: `stop_active=false`.

## 5. Passport / daily-report / quality-table

Финальные локальные артефакты:

- `.codex_local/wappi_controlled_watch/passport_20260627_final.json`
- `.codex_local/wappi_controlled_watch/daily_report_20260627_final.json`
- `.codex_local/wappi_controlled_watch/quality_table_20260627_final.csv`

Их не коммичу: quality-table содержит реальные тексты черновиков.

Финальный passport:

- `process_found=true`
- `pid=37750`
- `screen.detected=true`
- `repo.commit=7ab8c43`
- `profiles_count=4`
- `profile_channels={"telegram": 2, "max": 2}`
- `pairs_count=3`

Финальный daily-report:

- heartbeat свежий: `fresh=true`, `last_cycle_at=2026-06-27T15:53:57.167584+00:00`
- `draft_created=1`
- `notes_written=0`
- `errors=0`
- `pending_notes=0`
- `quarantined_pairs=0`
- `pair_missing=236` за окно 24ч
- resolver buckets: `amo_chat_event_* = 0`, `brand_mismatch=0`, `closed_lead=0`, `max_phone_missing=0`, `pending=0`, `quarantined=0`

Важное уточнение по цифрам:

- `pair_missing=236` включает строки, созданные до финального watermark-fix и во время первого sweep.
- После финального restart первый новый цикл: `processed=776`, `skipped=776`, `bot_calls=0`, `dry_run=true`.
- Следующий цикл: `processed=0`, `skipped=0`, `bot_calls=0`. Это подтверждает, что старый поток больше не повторяется.

Quality-table:

- строк: `6`
- route counts: `bot_answer_self_for_pilot=2`, `draft_for_manager=3`, `manager_only=1`
- все 6 строк имеют `manager_approval_required` и `no_auto_send`
- raw/service marker scan по `draft_text`: `0` hits

## 6. Safety / semantic status

Formal status: `PASS`.

Semantic status: `PASS_WITH_NOTES` только для запуска observe-контура, не для качества черновиков.

Что подтверждено:

- Клиенту ничего не отправляется.
- AMO/Tallanto/CRM write = 0.
- AMO notes = 0.
- Auto-resolver OFF.
- Bot-safe memory OFF.
- PID и env показываются из реального процесса.
- Observe-loop больше не раздувает journal одним и тем же старым потоком.

Что не подтверждено и идёт на регрейд:

- Польза черновиков для менеджеров.
- Wrong-brand/wrong-lead/опасные обещания по смыслу.
- Расширение на auto-resolver или AMO note write.

Следующий шаг:

1. Передать `quality_table_20260627_final.csv` Дмитрию/менеджерам на разметку.
2. Передать `passport_20260627_final.json`, `daily_report_20260627_final.json`, `quality_table_20260627_final.csv` Claude #1 на регрейд по сырью.
3. Не включать AMO note write, auto-resolver или клиентские отправки без отдельного явного «да».
