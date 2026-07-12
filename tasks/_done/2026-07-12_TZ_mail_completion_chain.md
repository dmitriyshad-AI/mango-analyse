> DONE 2026-07-12 20:48 | ветка codex/mail-three-stage-worktree-hygiene | codex

> TAKE 2026-07-12 20:27 | ветка codex/mail-three-stage-worktree-hygiene | codex

Ветка: codex/mail-three-stage-worktree-hygiene
Зоны: scripts/run_customer_timeline_mail_chain.py, scripts/run_customer_timeline_mail_download.py, scripts/run_customer_timeline_mail_process.py, scripts/run_customer_timeline_mail_import.py, deploy/customer_timeline_daily_captures/, tests/test_customer_timeline_mail_pipeline.py, docs/, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_customer_timeline_mail_pipeline.py tests/test_customer_timeline_codex_task.py
Семантический-аудит: нет

# Почта: последовательный запуск стадий по завершению

## Цель

Заменить три независимых запуска по часам одной цепочкой:
`скачал -> обработал -> влил`. Следующая стадия стартует только после успешного
завершения предыдущей.

## Требования

- переиспользовать существующие три стадии и их проверки;
- оставить один календарный запуск цепочки;
- занятый лок или неуспех стадии останавливает цепь с явной причиной;
- протухший манифест не позволяет стартовать следующей стадии;
- установка расписания остаётся отдельным ручным действием;
- не запускать живой IMAP и не писать в боевую timeline.

## СТОП

- следующая стадия стартует после неуспеха предыдущей;
- появляются параллельные писатели;
- тест вызывает живую почту или боевую базу.
