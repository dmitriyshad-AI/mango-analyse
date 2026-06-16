# SPIKE_REPORT TZ-131 Phase 0 C1-C4

Дата: 2026-06-16

Ограничение: спайки выполнялись только чтением. Целевые файлы `~/.codex/config.toml`, `~/.codex/AGENTS.md` и `stable_runtime/amocrm_runtime/codex_home/*` не менялись. При вызове `codex --help` сам CLI обновил служебные cache/state-файлы в `~/.codex`; это не изменение конфигурации проекта.

## C1. Профили Codex

Проверено: `codex --help`, `codex exec --help`, поиск по бинарю и `~/.codex`.

Вывод: профили берутся из `config.toml`, вероятный формат `[profiles.<name>]`. Отдельных файлов вида `~/.codex/*.config.toml` не найдено. Профиль может переопределять модель, personality и reasoning-настройки, но точный локальный TOML-пример не найден.

## C2. Hooks

Проверено: локальный README пакета, строки бинаря, пример `hooks.json` в установленном Figma plugin.

Вывод: hooks поддерживают события `PreToolUse`, `PermissionRequest`, `PostToolUse`, `SessionStart`, `UserPromptSubmit`, `Stop`. Это не только уведомления: `PreToolUse`/`PermissionRequest` могут блокировать действие, `PostToolUse` и `Stop` могут останавливать или продолжать исполнение. В текущих `~/.codex/config.toml` и `stable_runtime/.../config.toml` активных hook-секций нет.

## C3. Субагенты: model/reasoning

Проверено: локальные configs и описание `spawn_agent`.

Вывод: `multi_agent=true` включён в основном и runtime Codex home. При запуске субагента `model` и `reasoning_effort` можно задавать как override; если их не задавать, агент наследует родительскую модель/усилие. Для проекта правило остаётся: для сложных крупных задач до 6 субагентов, reasoning `xhigh`, без запуска субагентов для мелких правок.

## C4. Конфигурации

Проверено read-only:

- `~/.codex/config.toml`: `model="gpt-5.5"`, `model_reasoning_effort="xhigh"`, `personality="pragmatic"`, `approval_policy="never"`, `sandbox_mode="danger-full-access"`, `multi_agent=true`.
- `stable_runtime/amocrm_runtime/codex_home/config.toml`: `model="gpt-5.4"`, `model_reasoning_effort="xhigh"`, `personality="pragmatic"`, `approval_policy="never"`, `sandbox_mode="workspace-write"`, `network_access=true`, `multi_agent=true`.
- `stable_runtime/amocrm_runtime/codex_home/AGENTS.md`: пустой.
- `~/.codex/AGENTS.md`: содержит глобальные правила Дмитрия про русский язык, субагентов до 6, `xhigh`, semantic review и skill-venv.

Риск: скрипты `stable_runtime/start-*resolve-analyze*.sh` при создании временного Codex home копируют часть файлов из `~/.codex`, включая `config.toml`. Поэтому будущие изменения `~/.codex/config.toml` могут протечь в worker/runtime-контур при создании нового temp-home. В TZ-131 такие записи запрещены.

## Решение для TZ-131

- Step 0 из старого файла 03 про обновление CLI и запись в `~/.codex` не выполнять.
- Реализовать только репо-локальные инструменты: `project_now.py`, очередь задач, audit pack, preflight.
- `stable_runtime` и `~/.codex` остаются только источниками read-only сведений для отчёта.
