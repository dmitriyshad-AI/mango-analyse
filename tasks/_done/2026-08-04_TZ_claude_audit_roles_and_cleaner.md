> DONE 2026-08-04 17:35 | ветка codex/timeline-final-selective-integration-20260804 | codex

> TAKE 2026-08-04 17:24 | ветка codex/timeline-final-selective-integration-20260804 | codex

Ветка: codex/timeline-final-selective-integration-20260804
Зоны: .claude/agents/, .claude/skills/, docs/worktrees_registry.md, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest --collect-only -q tests/test_subscription_llm_draft_provider.py
Семантический-аудит: нет

# Короткие роли Claude для независимой приёмки

## Цель

Выборочно принять из `claude/timeline-final-20260803` только короткие роли
архитектора, ломателя и бизнес-аудитора, добавить постоянного уборщика и общий
процесс. Код бота, Timeline, оплаты и generated-артефакты не переносить.

## Приёмка

- четыре роли разбираются Claude CLI, работают в `permissionMode: plan` и имеют
  ограничение числа ходов;
- процесс требует инвентаризацию, сквозной тест, ломателя и чистый баланс строк;
- роль уборщика ничего не удаляет сама и отделяет доказанное удаление от риска;
- новых флагов, зависимостей и runtime-механизмов нет;
- донорский смысловой гейт личности отклонён по независимому BLOCK-аудиту.
- реестр содержит ровно фактические worktree из `git worktree list`.

## СТОП

- любая роль может менять файлы без отдельного подтверждения;
- переносится код бота, Timeline, оплат или generated-артефакт;
- процесс требует новый runtime-флаг, зависимость или второй механизм аудита.
