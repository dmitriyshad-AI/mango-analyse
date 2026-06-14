# TZ-103: судья действий, поставка 2

Дата: 2026-06-14  
Ветка/worktree: `codex/tz103-action-judge`, `/Users/dmitrijfabarisov/Projects/Mango_tz103_action_judge`

## Что сделано

- Перенесена база D6 из ТЗ-21 в текущий worktree: builder персон, D6 persona set, тесты и D6 отчёты.
- Шаг 0: словарь действий выровнен с боевым списком ТЗ-26: `book_trial` заменён на `send_materials`.
- Набор `product_data/telegram_dynamic_test_sets/autonomy_personas_unpk_20260613.jsonl` перематериализован из `deal_card` через обновлённое правило. Raw-доноры D6 в изолированном worktree отсутствовали, поэтому полный rebuild из `result.json` не выполнялся; `book_trial` по `scripts/`, `tests/`, `product_data/telegram_dynamic_test_sets` не остался.
- Добавлен детерминированный судья действий: `src/mango_mvp/channels/action_decision_judge.py`.
- Добавлен калибровочный CLI: `scripts/calibrate_action_decision_judge.py`.
- Добавлен ручной gold-набор 26 ходов: `product_data/telegram_dynamic_test_sets/action_decision_judge_gold_20260614.json`.
- Dynamic runner теперь пишет `expected_action`, `action_judge`, `action_decision` в JSONL/CSV/MD/summary и дампит фактический профиль флагов.

## Границы

- Боевой профиль не изменён: `TELEGRAM_DEAL_ACTION_DECISION` не добавлен в `pilot_gold_v1`.
- Включение action layer для прогона только через env: `TELEGRAM_DEAL_ACTION_DECISION=1`.
- В persona/context запрещены protection-flag keys; runner валится до прогона.
- Hard barriers считаются кодом, без модели: P0/срочная оплата, cross-brand, fabricated amount, action without precondition, missing action signal.
- Бренд и суммы для hard barriers берутся из `retrieved_facts`, не из текста ответа.
- Модельный text judge остаётся только для старого мягкого/текстового качества.

## Калибровка

Артефакт: `audits/_inbox/tz103_action_judge_calibration_20260614/calibration_report.json`

Итог:

```json
{"accepted": true, "total": 26, "unsafe_false_passes": 0, "hard_false_negatives": 0, "hard_false_positives": 0, "soft_false_positives": 1}
```

Покрытие gold-набора: оплата, CRM/card data, срочное+оплата, cross-brand, fabricated amount, send_materials, send_document, schedule, lead capture, follow-up, handoff.

## M1-прогон

Успешный содержательный прогон:

- Артефакт: `audits/_inbox/tz103_action_judge_m1_20260614_125652/`
- Machine: `arm64`
- Snapshot: `product_data/knowledge_base/kb_release_20260603_v6_5_summer_format_cleanup/kb_release_v3_snapshot.json`
- Env: `TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1`, `TELEGRAM_DEAL_ACTION_DECISION=1`
- CLI: `--parallel 4`, `--judge-prompt-version v9.1`, `--disable-bot-cache`

Summary:

- Dialogs: 12 completed
- Turns: 65
- Text judge: 11 `PASS_WITH_NOTES`, 1 `FAIL`
- Text hard gate: `p0_mishandled` in `autonomy_unpk_real_006`
- `action_decision`: 65/65 turns enabled
- `action_judge`: hard barriers `{}`, unsafe turns `0`
- Action reward: 6 turns
- Action accuracy over committed actions: `0.75`
- Action recall over expected actionable turns: `0.2222`

Infra attempts before success:

- `tz103_action_judge_m1_20260614_125408`: missing temp `CODEX_HOME`
- `tz103_action_judge_m1_20260614_125432`: temp `CODEX_HOME` without auth, 401
- `tz103_action_judge_m1_20260614_125616`: default `~/.codex/config.toml` had `service_tier="default"`, current CLI expects `fast|flex`
- Success used temp `CODEX_HOME` with copied auth/config and `service_tier="fast"`; user home config was not modified.

## Проверки

- `py_compile` по новым/изменённым Python-файлам: pass
- `ruff check --select F821 ...`: pass
- Focused pytest: `13 passed`
- Existing action/dynamic tests: `tests/test_deal_action_decision.py tests/test_manager_handoff_summary.py` -> `11 passed`
- Full dynamic runner tests: `tests/test_telegram_dynamic_client_sim.py` -> `101 passed`
- Full pytest: `3215 passed, 5 skipped, 1 warning`

## Остаточные риски

- `expected_action` в D6 persona set остаётся persona-level и не является ручной пометкой каждого turn; из-за этого recall на M1 трактовать осторожно. Ручная acceptance-калибровка находится в отдельном turn-level gold-наборе.
- Единственный M1 `FAIL` относится к старому text judge/P0 handling, не к hard barriers судьи действий.
- Claude #1 reggrade по сырью обязателен и не закрыт этим коммитом. Входные артефакты для регрейда: M1 out-dir и calibration report выше.

