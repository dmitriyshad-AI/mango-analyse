# TZ-131 operational foundation report

Дата: 2026-06-16

## Что сделано

- Фаза 0: создан `docs/codex_setup/SPIKE_REPORT_2026-06-16.md` по C1-C4. Целевые `~/.codex/config.toml`, `~/.codex/AGENTS.md` и `stable_runtime/.../codex_home/*` не менялись; при read-only вызове `codex --help` сам CLI обновил служебный state/cache в `~/.codex`, это отдельно зафиксировано в SPIKE_REPORT.
- Шаг 1: добавлен `scripts/project_now.py`, `docs/BLOCKERS.yaml`, `docs/PROJECT_NOW.md` добавлен в `.gitignore`; из `AGENTS.md` убран протухший блок Current Main Priority, добавлены PROJECT_NOW, очередь, preflight, интерфейсы и шапка ТЗ.
- Шаг 2: добавлены `scripts/task_move.py`, `scripts/task_stale_report.py`, созданы `tasks/_running/.gitkeep` и `tasks/_failed/.gitkeep`. В `_running` перенесён только TZ-131; старый inbox не трогался.
- Шаг 3: добавлен `scripts/make_audit_pack.py`; ПДн-фильтр маскирует телефоны РФ и email. CLIENT_PATHS сверены по факту репо: `product_data/knowledge_base/`, `src/mango_mvp/channels/`, `src/mango_mvp/integrations/draft_loop.py`, `scripts/run_amo_wappi_draft_loop.py`, `scripts/run_telegram_public_pilot_bots.py`, `product_data/telegram_dynamic_test_sets/`.
- Шаг 4: добавлен `scripts/preflight.py`; закрыты TODO: запретные зоны, безопасный `--collect-only`, реестр worktree, игнор `prunable`/`detached`/`locked`. Дополнительно учитывается worktree, зарегистрированный по branch-name.

## Проверки

- `python3 scripts/project_now.py` — OK.
- `python3 scripts/task_stale_report.py` — OK, отчёт: `docs/_automation_status/stale_tasks.md`.
- `python3 scripts/preflight.py --tz tasks/_running/2026-06-16_TZ131_etap1_operacionnyy_fundament.md` — OK.
- Тест-команда из шапки: `12 passed in 0.09s`.
- Полный pytest: `3328 passed, 2 skipped, 1 warning in 50.75s`.

## Audit pack

- `audits/_inbox/tz131_operational_foundation_20260616232919`

## Остаточные риски

- `CURRENT_STATE.md` и `ROADMAP.md` остаются историческими; текущий источник состояния теперь `docs/PROJECT_NOW.md`.
- ФИО в audit pack не маскируются автоматически по решению ТЗ-131, чтобы не ловить шум; для клиентских/CRM артефактов остаётся ручной semantic/manual gate.
