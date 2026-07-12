> DONE 2026-07-12 17:27 | ветка codex/mail-three-stage-worktree-hygiene | codex

> TAKE 2026-07-12 16:12 | ветка codex/mail-three-stage-worktree-hygiene | codex

Ветка: codex/mail-three-stage-worktree-hygiene
Зоны: src/mango_mvp/productization/mail_archive.py, scripts/mango_office_mail_archive.py, scripts/build_customer_timeline_nightly_dv2_sources.py, scripts/run_customer_timeline_codex_task.py, scripts/run_customer_timeline_nightly_incremental.py, scripts/run_customer_timeline_mail_download.py, scripts/run_customer_timeline_mail_process.py, scripts/run_customer_timeline_mail_import.py, deploy/customer_timeline_daily_captures/, tests/, docs/DECISIONS_LOG.md, docs/worktrees_registry.md, AGENTS.md, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_productization_mail_archive.py tests/test_customer_timeline_mail_pipeline.py tests/test_customer_timeline_nightly_incremental.py tests/test_customer_timeline_codex_task.py
Семантический-аудит: нет

# ТЗ-Г: почтовый конвейер из трёх стадий и порядок worktree

## Цель

Собрать уже существующие компоненты в независимый почтовый конвейер:

1. скачать почту через существующий IMAP-архиватор только на чтение;
2. обработать архив через существующий `build_mail_increment()`;
3. импортировать готовый приращённый набор через существующий `nightly_incremental`.

Каждая стадия имеет отдельный безопасный манифест, курсор, блокировку и ненулевой код возврата при неполном результате. Повторный запуск не создаёт дублей.

## Границы

- Не писать в AMO, Tallanto, CRM и клиентские каналы.
- Не менять боевую `customer_timeline`; ручные проверки только на SQLite-копии.
- Не печатать адреса почты, пароли, токены и содержимое писем в отчёты.
- Использовать только `~/.mango_secrets/mail_imap_edu_kmipt.env`, значения не копировать.
- Загружать только `INBOX` и `Sent`; черновики не являются историей клиента.
- Не строить второй IMAP-загрузчик, второй парсер писем или второй импортёр.
- Не заявлять точный IMAP UID-курсор: на первом этапе используется честный overlap-waterline + SHA-дедупликация.
- Не устанавливать и не перезапускать `launchd` без отдельного подтверждения Дмитрия после трёх ручных циклов.
- Worktree не удалять без отдельного точного предполёта и подтверждения.

## Этап A: скачивание

- Тонкая обёртка вызывает `scripts/mango_office_mail_archive.py ingest`.
- Постоянные каталоги по ящикам: `incoming/regru_edu/inbox` и `incoming/regru_edu/sent`.
- Единая блокировка не допускает параллельные запуски.
- По умолчанию нет искусственного лимита писем; ограниченный запуск с усечением считается ошибкой.
- Манифест хранит только безопасные счётчики и технические метки.
- Курсор обновляется атомарно только после полного успеха обоих ящиков.

## Этап B: обработка

- Использовать `build_mail_increment()` и существующие правила ПДн/дедупликации.
- Читать основную каноническую БД и найденные `incoming/**/mail_archive.sqlite`.
- Нижняя граница берётся из `ingestion_cursors.mail_archive_stage2` тестовой timeline с overlap 300 секунд.
- Отсутствующий курсор требует явного bootstrap-режима; молча импортировать всю историю нельзя.
- Запуск разрешён только после свежего успешного манифеста Этапа A.
- Результат: JSONL, манифест и конфигурация для Этапа C.

## Этап C: импорт

- Вызвать существующий `scripts/run_customer_timeline_nightly_incremental.py`.
- Запуск разрешён только после свежего успешного манифеста Этапа B.
- Любой `gate_passed=false`, обязательный источник с ошибкой или неполный результат дают ненулевой код возврата.
- Курсор сохраняется только после успешного импорта.

## Расписание

Подготовить, но не устанавливать, три `launchd`-шаблона с одним корневым каталогом и последовательными окнами: скачивание, обработка, импорт. Следующая стадия обязана проверять свежесть манифеста предыдущей.

## Журнал решений и worktree

- Свести `docs/DECISIONS_LOG.md` смысловым объединением: одинаковый номер не может означать разные решения.
- Сохранить все уникальные решения, конфликтующие перенумеровать с указанием источника.
- Добавить правило: один активный worktree — один исполнитель/одна задача; чужие runtime-каталоги и `.codex_local` не удалять.
- Обновить реестр worktree по фактическому `git worktree list`; активные деревья не снимать.

## Приёмка

1. Точечные и полный `pytest` зелёные.
2. Три ручных цикла на SQLite-копии: второй и третий не создают дублей.
3. Ошибка/усечение любой стадии останавливает следующую и возвращает ненулевой код.
4. В логах и манифестах нет секретов и содержимого писем.
5. Создан audit pack с командами, результатами и явным перечнем того, что не менялось live.
6. Независимый аудитор проверяет каждый этап по сырому коду и артефактам.
