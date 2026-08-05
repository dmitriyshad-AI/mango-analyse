> DONE 2026-08-05 15:18 | ветка codex/global-code-cut-wave11 | codex

> TAKE 2026-08-05 15:12 | ветка codex/global-code-cut-wave11 | codex

Ветка: codex/global-code-cut-wave11
Зоны: scripts/, docs/, tests/, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest --collect-only -q
Семантический-аудит: нет

# Уборка wave 11: одноразовые archive/history-скрипты

## Проблема

В `scripts/` остаются восемь одноразовых программ старых волн обработки архивов
и истории звонков. Они не входят в текущие entrypoints, службы или task queue,
не импортируются кодом и дублируют закрытые операции, результат которых уже
поглощён каноническими calls/mail/Customer Timeline контурами.

## Образ результата и бизнес-польза

- Удалено 1540 строк неиспользуемого исполняемого кода и устаревшие строки
  safety-матрицы.
- Текущие calls, mail, Customer Timeline, bot draft и CRM read-only пути не
  меняются.
- Новых файлов, функций, флагов и зависимостей нет.
- Репозиторий становится проще: инженер не может случайно запустить старую
  destructive обработку вместо канонического процесса.

## Доказательство до удаления

1. Свежий Graphify используется как навигация; отсутствие проверяется `git grep`
   и чтением исходников на текущем HEAD.
2. Для каждого файла отсутствуют импорты, shell/launchd/cron/workflow entrypoints,
   активные ТЗ и тесты.
3. Каждый файл является самостоятельным `main()` старой датированной операции и
   не предоставляет библиотечный API текущему коду.

## Удалить

- `scripts/audit_local_archive_messages1_zip.py`
- `scripts/prepare_contact_history_batch.py`
- `scripts/prepare_history_gap_wave.py`
- `scripts/prepare_message_archive_history_full_cycle.py`
- `scripts/prepare_message_archive_wave.py`
- `scripts/prepare_message_archives_history_full_cycle.py`
- `scripts/prepare_phone_history_batch.py`
- `scripts/repair_and_move_message_archives.py`
- только соответствующие записи и обобщающую ссылку в `docs/SCRIPT_SAFETY_MATRIX.md`

## Приёмка

1. `git grep` не находит имён удалённых файлов в текущих документах, коде,
   тестах, active tasks, launchd или workflows.
2. `pytest --collect-only` не теряет тесты из-за импортов.
3. Точечные Graphify/ops тесты и полный pytest не получают новых падений.
4. Diff содержит только удаления, task move и обновление реестра worktree.

## Ограничения

- Не удалять raw/runtime/архивы, БД и результаты обработки.
- Не трогать живые calls/mail/Timeline/бот модули.
- Не восстанавливать функциональность удаляемых скриптов новым фасадом.

## СТОП

- Найден текущий импорт, entrypoint, служба, активное ТЗ или тест, зависящий от
  любого кандидата.
- Удаление требует замены новым кодом либо изменяет клиентский/live/runtime путь.
- Рабочее дерево получает чужие изменения в заявленных зонах.
