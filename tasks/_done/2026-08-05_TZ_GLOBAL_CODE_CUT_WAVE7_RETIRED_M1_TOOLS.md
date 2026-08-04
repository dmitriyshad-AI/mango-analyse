> DONE 2026-08-05 02:17 | ветка main | codex

> TAKE 2026-08-05 02:07 | ветка main | codex

Ветка: main
Зоны: scripts/enrich_adr003_existing_frame_proof_shadow.py, tests/test_enrich_adr003_existing_frame_proof_shadow.py, scripts/build_marathon2_m1_bundles.py, scripts/build_memory_measure_scenarios.py, scripts/probe_memory_measure_context.py, scripts/run_memory_measure_off_on.py, tests/test_marathon2_m1_bundles.py, tests/test_memory_measure_apparatus.py, product_data/telegram_dynamic_test_sets/memory_rich_2026-06-21.jsonl, scripts/run_m1_mail_summary_merge.py, scripts/run_marathon2_mail_summary_enrich.py, tests/test_run_m1_mail_summary_merge.py, tests/test_marathon2_mail_summary_enrich.py, scripts/build_marathon2_transfer_package.py, tests/test_marathon2_transfer_package.py, tasks/, audits/_inbox/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_report_adr003_semantic_frame_eval.py tests/test_wappi_replay_runner.py tests/test_customer_timeline_nightly_service.py tests/test_publish_snapshot_tooling.py
Семантический-аудит: нет

# Уборка заменённых M1/F5/mail/SWAP инструментов

## Образ результата

В репозитории остаются по одному текущему владельцу:

- ADR-003: канонический semantic-frame отчёт;
- память Wappi: `replay_exam` с ON-only режимом;
- почта Timeline: трёхстадийный nightly D-091;
- публикация Timeline: `scripts/publish_snapshot/` D-064.

Старые разовые OFF/ON, mail merge/enrich, proof-shadow и transfer/SWAP
инструменты удалены вместе с собственными тестами и одним старым набором.

## Доказательство до правок

- Graphify и сырой поиск на `cb1e83d1`: живых вызывающих нет.
- LaunchAgent, deploy, CLI, текущие ТЗ и RUNBOOK не ссылаются на кандидатов.
- D-090 запрещает старый полный OFF/ON; D-091 задаёт новый mail-путь;
  D-064 задаёт новый publish/rollback-путь.
- Исторические D-043/D-047/D-053 описывают старое поколение и остаются в журнале.
- Все 15 файлов имеют текущего преемника; нового кода не требуется.

## Минимальное решение

Удалить ровно 15 файлов и 6493 строки. Не переносить helper-функции: у них нет
текущего потребителя, а восстановление доступно через Git.

## Приёмка

1. `15 D`, `6493` удалённых строк, добавлено 0 строк кода.
2. После удаления нет текущих ссылок.
3. Полный collect уменьшается только на тесты кандидатов.
4. Канонические ADR/P0/Wappi/nightly/publish тесты зелёные.
5. Полный pytest не получает новых падений относительно wave 6.
6. Независимый ломатель подтверждает владельцев-преемников.

## СТОП

- Найден текущий потребитель, активная задача или служба.
- Уникальная функция нужна текущему владельцу.
- Удаление требует правки живого `src/`.

## Результат 2026-08-05

- Удалено: 15 файлов, 6493 строки.
- Добавлено кода: 0; новых файлов кода, флагов и зависимостей: 0.
- Текущих runtime/deploy/launchd/task/RUNBOOK потребителей: 0.
- Collect: 5207, ровно минус 57 тестов кандидатов.
- Survivor-набор: 199 passed.
- Полный pytest: 5196 passed, те же 8 KB baseline failures, 3 skipped.
- Claude Fable: ACCEPT CLEAN.
- Независимый Codex-ломатель: ACCEPT.
- D-043/D-047/D-053 сохранены как история; их актуальные функции принадлежат
  D-064/D-090/D-091.
- Runtime, базы и внешние системы не менялись.
- Варианты «оставить» и «перенести helpers» отвергнуты: поддерживали бы вторые
  исполняемые пути рядом с уже принятыми владельцами.
