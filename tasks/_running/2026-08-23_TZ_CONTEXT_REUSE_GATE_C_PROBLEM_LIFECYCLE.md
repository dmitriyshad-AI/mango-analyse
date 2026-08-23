> TAKE 2026-08-23 17:07 | ветка codex/context-reuse-gate-20260823 | codex

# ТЗ C: попытка не равна закрытой проблеме

Ветка: codex/context-reuse-gate-20260823
Зоны: scripts/task_move.py, scripts/project_now.py, tests/test_task_move.py, tests/test_project_now.py, AGENTS.md, docs/, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_task_move.py tests/test_project_now.py
Семантический-аудит: нет
Feature-ID: process.context_reuse_gate.problem_lifecycle_v1
Problem-ID: process.context_loss_and_duplicate_builds
Изменение: extend
Ключевые-символы: task_move,project_now,Problem-ID,Исход
Ключевые-слова: attempt_complete,problem_closed,legacy_unknown,Closure-evidence

Дата: 2026-08-23.
Предусловие: ТЗ A1, A2 и B приняты в интеграционной ветке.

## Контекст

Сейчас `task_move.py` переносит файл в `_done`, но перенос не доказывает, что
исходная проблема закрыта. Старые ТЗ массово не переписывать и отдельный
ручной feature-ledger не создавать.

## Реализация

1. Для новых ТЗ с `Problem-ID` добавить в `task_move.py` аргумент:
   `--outcome attempt_complete|problem_closed|blocked|superseded`.
2. Перед переносом записывать или обновлять единственное поле `Исход:` в ТЗ.
3. `problem_closed` разрешать только с непустым `Closure-evidence:`.
4. Для legacy-ТЗ без `Problem-ID` сохранить текущую совместимость и показывать
   `legacy_unknown`.
   Для `superseded` продолжение указывать существующим полем
   `Следующий шаг:`, не создавать второй реестр.
5. `project_now.py` генерирует по заголовкам задач:
   - открытые Problem-ID;
   - активную попытку и worktree/HEAD;
   - последнюю завершённую попытку;
   - её исход;
   - следующий шаг;
   - конфликт двух активных Feature-ID.
6. Источником остаются сами ТЗ/Git/audit evidence; отдельную БД, YAML или
   редактируемый ledger не добавлять.

## СТОП

- требуется массовая миграция старых 300+ ТЗ;
- создаётся второй ручной источник истины;
- `_done` автоматически трактуется как `problem_closed`;
- task_move удаляет, переписывает или переносит чужую задачу без явной команды;
- бюджет более 150 строк нетестового кода.

## Приёмка

1. Тест-команда зелёная.
2. `attempt_complete` остаётся открытой проблемой в PROJECT_NOW.
3. `problem_closed` без evidence блокируется.
4. `superseded` указывает продолжение, не закрывая проблему.
5. Две активные задачи одного Feature-ID видны как конфликт.
6. Legacy-задачи продолжают переноситься и помечаются `legacy_unknown`.
7. Новых сервисов, зависимостей и флагов нет.
