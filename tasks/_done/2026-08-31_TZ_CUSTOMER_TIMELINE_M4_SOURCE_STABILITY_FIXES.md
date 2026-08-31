> DONE 2026-08-31 15:32 | ветка codex/customer-timeline-runtime-20260830 | codex

> TAKE 2026-08-31 13:55 | ветка codex/customer-timeline-runtime-20260830 | codex

Ветка: codex/customer-timeline-runtime-20260830
Зоны: src/mango_mvp/integrations/draft_loop.py, src/mango_mvp/customer_timeline/wappi_history_import.py, src/mango_mvp/customer_timeline/nightly_incremental.py, src/mango_mvp/customer_timeline/nightly_service.py, src/mango_mvp/productization/mail_imap_snapshot.py, src/mango_mvp/productization/mail_archive.py, tests/test_draft_loop.py, tests/test_wappi_history_import_to_timeline.py, tests/test_productization_mail_archive.py, tests/test_customer_timeline_nightly_incremental.py, tests/test_customer_timeline_nightly_service.py, tasks/, audits/_inbox/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_draft_loop.py tests/test_wappi_history_import_to_timeline.py tests/test_productization_mail_archive.py tests/test_customer_timeline_nightly_service.py
Семантический-аудит: да
Feature-ID: feature.customer_timeline.m4_source_stability
Problem-ID: problem.customer_timeline.mutable_wappi_input_and_imap_disconnect
Исход: problem_closed
Closure-evidence: audits/_inbox/customer_timeline_m4_source_stability_20260831T105541Z; 5664 full-suite passed before final safety hardening; 462 relevant passed after all fixes; independent architect, breaker and business auditor final P0/P1=0 and semantic PASS
Изменение: fix
Ключевые-символы: load_pairs_file,load_wappi_pairs,run_wappi_history_import,build_mail_archive_ingest,open_imap_client_with_retries
Ключевые-слова: immutable file descriptor snapshot,atomic replace provenance drift,IMAP reconnect,resume mailbox tail

# Customer Timeline M4: стабильные входы Wappi и почты
## Проблема

Первый M4 warehouse-run на чистом M1 seed не прошёл:

1. Живой draft-loop атомарно заменил auto-pairs во время долгого Wappi-read.
   Импортёр уже загрузил одну версию в память, но сравнил SHA пути в конце и
   ложно объявил provenance drift; 354 новых сообщения не были записаны.
2. IMAP разорвал read-only соединение после части трёхдневного окна. Один
   reconnect также попал в краткий DNS-сбой, после чего импорт остановил
   mailbox и при повторе снова начал окно сначала.

## Сделать

1. Переиспользовать verified-file-descriptor паттерн проекта: пары Wappi
   разобрать из однократно прочитанных байтов и привязать provenance к SHA
   именно этих байтов. Последующую атомарную замену пути показывать
   диагностикой, но не считать дрейфом уже закреплённого входа.
2. Следующий run обязан прочитать уже новую версию auto-pairs.
3. Для transient IMAP fetch/reconnect сделать ограниченное число повторов с
   паузой; читать и повторять по UID, чтобы reconnect/EXPUNGE не смещал
   выбранные письма. После успешного reconnect продолжить тот же список UID.
   Постоянная ошибка остаётся красной и попадает в report.
4. Не менять P0/brand/bot visibility, не писать в prod/CRM/Tallanto, не
   запускать LLM/ASR и не ослаблять provenance других входов.

## СТОП

- Нельзя доказать, какие именно байты auto-pairs были разобраны импортёром.
- Для восстановления IMAP потребовалось бы пропустить письмо или признать
  неполный mailbox зелёным.
- Изменение затрагивает live-write, клиентские отправки или внешние записи.

## Приёмка

- atomic replace auto-pairs во время run не даёт provenance_drift;
- изменение importer/phase config по-прежнему блокирует apply;
- следующий run видит новую auto-pairs версию;
- два последовательных transient IMAP disconnect восстанавливаются без
  пропуска хвоста и без дублей;
- изменение порядковых номеров писем после reconnect не меняет выбранные UID;
- исчерпание bounded retries остаётся ошибкой;
- целевые тесты и полный релевантный набор зелёные;
- независимый breaker и architect-auditor не находят P0/P1.
