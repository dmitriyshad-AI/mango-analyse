# TZ-22 D7 A2: CRM service marker gate

Дата: 2026-06-14

## Что сделано

- Добавлен `SERVICE_TEST_MARKERS_RE` в `crm_text_quality_detector.py`.
- Добавлен детект `service_test_marker` с `severity="P0"` и `class_id="Q-service-marker"`.
- Детект намеренно проходит строго по `TARGET_CRM_TEXT_FIELDS`, не через `_iter_target_text_fields`, чтобы не блокировать ручное поле `История общения`.
- Новый P0-риск добавлен в blocking/fail-live списки `scripts/run_crm_writeback_quality_gate.py`.
- Добавлены NEG-тесты на ложные срабатывания:
  - живое слово `тестов`;
  - ручное поле `История общения`;
  - ручное поле с настоящим marker-текстом;
  - авто-поле с `Тестовая история`.

## Границы

- Live AMO/Tallanto write не запускался.
- ASR/analyze/Resolve+Analyze не запускались.
- `stable_runtime` не менялся.
- Проверка 50 строк снимка `tz14_amo_step1_full_20260612` не выполнялась: большие ignored-снимки отсутствуют в worktree, по решению Дмитрия сырьевой регрейд делает архитектор.
