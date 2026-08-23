> TAKE 2026-08-24 00:16 | ветка codex/m1-calls-provider-selective-integration-20260824 | codex

# ТЗ: восстановить обязательный preflight на M1

Ветка: codex/m1-calls-provider-selective-integration-20260824
Зоны: scripts/make_audit_pack.py, tests/test_audit_pack_pii.py, docs/worktrees_registry.md, tasks/, audits/_inbox/m1_calls_preflight_process_repair_20260824/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_audit_pack_pii.py tests/test_preflight.py
Семантический-аудит: нет
Feature-ID: process.m1_calls_preflight_repair.v1
Problem-ID: process.m1_calls_preflight_blocked_by_none_and_registry
Изменение: extend
Ключевые-символы: _review_prompt,run_claude_review,parse_worktrees_porcelain
Ключевые-слова: SELECTED_OWNER NONE,M1 worktree registry,Claude receipt

Дата: 2026-08-24.

## Цель

Восстановить обязательный процесс проверки перед новой офлайн-интеграцией
Calls. Это узкий ремонт существующих механизмов, а не новый preflight.

## Подтверждённые причины остановки

1. Если inventory не выбрал владельца, валидатор ожидает строку
   `SELECTED_OWNER: NONE`, но сгенерированный запрос Claude не говорит явно,
   что при пустом `selected_owner.path` нужно написать именно `NONE`.
2. Локальный реестр не содержит существующие веточные worktree M1, поэтому
   обязательная проверка останавливается до анализа продуктового кода.
3. В продуктовом ТЗ входное поле должно быть `Изменение: extend`; решение
   `port` разрешено только как результат inventory.

## Разрешённые изменения

- Явно записать в Claude prompt значение владельца, вычисленное из inventory:
  путь либо `NONE`.
- Добавить регрессионный тест для пустого `selected_owner`.
- Зарегистрировать существующие веточные worktree M1 без удаления, переключения
  веток и изменения их содержимого.
- После ремонта сформировать новое продуктовое ТЗ с `Изменение: extend` и
  повторить полный preflight.

## Запрещено

- Запускать production-службу, Mango API, ASR, Resolve, Analyze и внешние записи.
- Удалять или чистить worktree.
- Считать этот ремонт успешным preflight продуктовой интеграции.
- Вызывать Claude до исправления формата `NONE` и повторного формирования пакета.

## Приёмка

1. Prompt содержит точную строку `SELECTED_OWNER: NONE`, если владельца нет.
2. Существующий вариант с путём владельца не изменил поведение.
3. Целевые тесты проходят.
4. Все существующие веточные worktree, кроме текущего, проходят проверку реестра.
5. Продуктовый preflight запускается заново на новом пакете и новом inventory.

## СТОП

- Любая необходимость включить рабочую службу или записать данные наружу.
- Неясное состояние или утрата уникальных изменений любого worktree.
- Невозможность подтвердить ремонт автоматическим тестом.
