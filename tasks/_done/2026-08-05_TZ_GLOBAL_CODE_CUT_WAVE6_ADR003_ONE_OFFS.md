> DONE 2026-08-05 02:04 | ветка main | codex

> TAKE 2026-08-05 01:48 | ветка main | codex

Ветка: main
Зоны: scripts/report_adr003_action_gate_counterfactual_proof.py, scripts/report_adr003_current_handoff_fact_gap.py, scripts/report_adr003_exact_proof_injection_shadow.py, scripts/report_adr003_existence_fact_verification.py, scripts/report_adr003_fact_gated_self_answer_readiness.py, scripts/report_adr003_frame_calibration_queue.py, scripts/report_adr003_manager_only_exact_proof_root_cause.py, scripts/report_adr003_overhandoff_levers.py, scripts/report_adr003_partial_answer_opportunities.py, scripts/report_adr003_partial_answer_policy_shadow.py, scripts/report_adr003_source_axis_blockers.py, scripts/report_adr003_source_axis_root_causes.py, tests/test_report_adr003_action_gate_counterfactual_proof.py, tests/test_report_adr003_current_handoff_fact_gap.py, tests/test_report_adr003_exact_proof_injection_shadow.py, tests/test_report_adr003_existence_fact_verification.py, tests/test_report_adr003_fact_gated_self_answer_readiness.py, tests/test_report_adr003_frame_calibration_queue.py, tests/test_report_adr003_manager_only_exact_proof_root_cause.py, tests/test_report_adr003_overhandoff_levers.py, tests/test_report_adr003_partial_answer_opportunities.py, tests/test_report_adr003_partial_answer_policy_shadow.py, tests/test_report_adr003_source_axis_blockers.py, tests/test_report_adr003_source_axis_root_causes.py, tasks/, audits/_inbox/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_report_adr003_semantic_frame_eval.py tests/test_subscription_llm_draft_provider.py tests/test_output_verification_floor_contract.py
Семантический-аудит: нет

# Уборка одноразовых отчётов ADR-003

## Образ результата

В репозитории остаются канонический отчёт ADR-003, текущий M1-экзамен и живой
direct-path. Двенадцать завершённых диагностических цепочек F2 удалены вместе
с тестами, которые проверяли только эти одноразовые отчёты.

## Доказательство до правок

- Graphify используется только как навигация; отсутствие подтверждается поиском
  по текущему исходному дереву.
- У кандидатов нет вызывающих вне собственных скриптов и тестов, завершённых ТЗ
  и архивных документов.
- В launchd/deploy, Wappi draft-loop, P0 evaluator, Customer Timeline и активных
  ТЗ ссылок нет.
- Текущие владельцы не удаляются:
  `scripts/report_adr003_semantic_frame_eval.py`,
  `scripts/run_p0_model_led_m1_eval.py` и
  `src/mango_mvp/channels/subscription_llm_parts/`.

## Минимальное решение

Удалить только 24 доказанно замкнутых файла. Новый фасад, архивная копия,
feature flag или замена не нужны: история уже сохранена в Git и тегах.

## Приёмка

1. Удалено ровно 24 файла и 9 837 строки, добавлено 0 строк рабочего кода.
2. Поиск текущих ссылок после удаления пуст.
3. Сбор тестов проходит.
4. Канонический ADR-003, P0/output-floor и Wappi business-journey тесты зелёные.
5. Полный pytest не получает новых падений.
6. Ломатель подтверждает, что живой путь и текущий измеритель не удалены.

## СТОП

- Найден реальный запускатель или активное ТЗ, использующее кандидата.
- Удаление требует изменения живого кода.
- Появляется новое падение вне удалённых самотестов.

## Результат 2026-08-05

- Удалено ровно 24 файла и 9837 строк; добавлено 0 строк кода.
- Новых файлов рабочего кода, флагов и зависимостей: 0.
- Текущих ссылок после удаления: 0.
- Полный collect: 5264 теста.
- Survivor/business-набор: 383 passed.
- Import-bomb ломателя: 572 passed, обращений к удалённым модулям 0.
- Полный pytest: 5253 passed, 8 известных KB baseline failures, 3 skipped.
- Независимый Codex-ломатель: ACCEPT.
- Claude CLI Fable 5 не нашёл живого потребителя или текущего уникального
  расчёта; его запрошенные проверки выполнены основным Codex и ломателем.
- Runtime, базы, AMO, Tallanto, CRM и Wappi не менялись.
- Более простые варианты «оставить» и «перенести helpers» отвергнуты:
  сохраняли бы закрытую диагностическую архитектуру без потребителя.
