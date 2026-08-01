> DONE 2026-07-31 05:47 | ветка codex/calls-dialogue-m1-20260730 | codex

> TAKE 2026-07-31 04:12 | ветка codex/calls-dialogue-m1-20260730 | codex

Ветка: codex/calls-dialogue-m1-20260730
Зоны: src/mango_mvp/customer_timeline/calls_two_processes.py, scripts/pull_mango_calls_drop_remote.py, scripts/receive_mango_calls_drop.py, scripts/run_mango_calls_process.sh, scripts/install_mango_calls_two_processes_service.py, scripts/publish_daily_mango_calls_google.py, scripts/export_daily_mango_calls_resolve.py, scripts/mango_calls_readonly_rsync_gate.sh, tests/test_mango_calls_remote_handoff.py, tests/test_mango_calls_schedule.py, tests/test_mango_calls_two_processes.py, tests/test_publish_daily_mango_calls_google.py, tests/test_export_daily_mango_calls_resolve.py, docs/, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_mango_calls_schedule.py
Семантический-аудит: да

# ТЗ: безопасный перенос Process A на M1

## Цель

Process A работает на M1; основной Mac сам забирает только запечатанный ready-drop
по прямому read-only SSH, а Process B остаётся рядом с актуальной Timeline staging.

## Требования

1. Получатель дважды забирает manifest вокруг SQLite и проверяет неизменность,
   SHA-256, размер и quick_check SQLite.
2. Только явный execute/confirmation; dry-run без сети и записи.
3. Передача только read-only SSH/rsync в owner-only incoming, никогда через Яндекс Диск;
   M1 не получает права записи или исполнения команд на основном Mac.
4. Приёмник повторно проверяет пакет и атомарно заменяет DB сначала, manifest последним.
5. Повтор того же SHA идемпотентен; неоднозначный/повреждённый пакет блокируется.
6. Приёмник хранит один локальный rollback hardlink предыдущего drop.
7. Process B запускается только после успешной приёмки; Process A на M1 не запускает локальный B.
8. Локальный режим A+B остаётся неизменным без полного remote config.
9. В логах нет телефонов, ФИО, email, строк расшифровки или секретов.
10. Runbook фиксирует preflight, direct rsync -H runtime, exact SHA, rollback и
    запрет cutover при активном Process A/claims.
11. На M1 launchd устанавливает только Process A; Process B там не загружается.

## Приёмка

- синтетический valid package принимается и проходит quick_check;
- tampered DB/manifest/path/confirmation блокируются;
- DB/manifest swap order и rollback проверены;
- same SHA reused, а Process B выполняется синхронно под общей блокировкой;
- puller строит read-only SSH/rsync без shell-инъекции и не делает сеть в dry-run;
- wrapper local/remote modes покрыты тестами;
- независимый breaker audit не находит P0/P1.

## СТОП

- Не останавливать текущие службы.
- Не запускать ASR/R+A, Process A/B, Mango/Tallanto API.
- Не писать в реальный runtime, M1, Timeline, Google, Яндекс, AMO/CRM.
- Не читать и не копировать секреты.

## Бритва

Переиспользовать текущие sealed manifest и ready_drop_fingerprint; не строить
новую очередь, брокер или сервер. Новые файлы обоснованы разными границами
доверия: сетевой puller не должен смешиваться с атомарной локальной приёмкой.
