> DONE 2026-07-02 16:45 | ветка codex/adr003-semanticframe-migration | codex

Ветка: codex/adr003-semanticframe-migration
Зоны: scripts/report_adr003_frame_calibration_queue.py, tests/test_report_adr003_frame_calibration_queue.py, tasks/_done/2026-07-02_TZ_ADR003_F2u_fact_gated_queue_summary_dlya_D1.md, audits/_inbox/adr003_f2u_fact_gated_summary_20260702164500/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_report_adr003_frame_calibration_queue.py tests/test_report_adr003_fact_gated_self_answer_readiness.py tests/test_report_adr003_existence_fact_verification.py tests/test_report_adr003_manager_only_exact_proof_root_cause.py
Семантический-аудит: да

# ADR-003 F2u: fact-gated summary in calibration queue

## Контекст

Claude #1 поправил гипотезу F2b: свежий M1 `36ea110` показывает реальный
over-handoff не в harmless ack/status, а в путанице "существует ли
курс/лагерь/формат для класса X" против `check_availability`/`enroll`.

Код уже имел отдельные read-only отчеты:

- `report_adr003_existence_fact_verification.py`;
- `report_adr003_fact_gated_self_answer_readiness.py`;
- `report_adr003_manager_only_exact_proof_root_cause.py`;
- `report_adr003_frame_calibration_queue.py`.

Проблема: общий calibration queue не выводил явные итоги fact-gated readiness,
поэтому Claude #1 приходилось собирать картину из нескольких отчетов.

## Цель

Сделать маленькое report-only расширение: общий `report_adr003_frame_calibration_queue.py`
подтягивает итоги `fact_gated_self_answer_readiness` и показывает их в JSON/Markdown.

Поведение бота не меняется.

## Что изменено

- Добавлен вызов `build_fact_gated_report(...)` в общий calibration queue.
- В `totals` добавлены численные поля:
  - `fact_gated_strict_f3_draft_candidates`;
  - `fact_gated_manager_only_exact_proof_needs_policy`;
  - `fact_gated_already_self_exact_proof`;
  - `fact_gated_blocked_no_exact_proof`;
  - `fact_gated_excluded_danger_money_p0`;
  - `fact_gated_current_handoff_rows`.
- В Markdown добавлен раздел `Fact-Gated Readiness Summary`.
- Тест закрепляет, что manager-only exact-proof строка видна в новом summary.

## Сырой результат на свежем M1 36ea110

Вход:

`/Users/dmitrijfabarisov/Yandex.Disk.localized/OpenClaw/Actual Mango Tests/adr003_f2_clean_36ea110_20260702/runs/adr003_f2_self_answer_shadow_36ea110/ON/dynamic_dialog_transcripts.jsonl`

Отчет:

`audits/_inbox/adr003_f2u_fact_gated_summary_20260702164500/reports_36ea110/adr003_frame_calibration_queue_report.json`

Итоги:

- `true_frame_too_cautious=14`;
- `stable_existence_as_check_availability=7`;
- `stable_existence_as_enroll=1`;
- `clean_route_only_discussion=0`;
- `factless_ack_status=0`;
- `fact_gated_strict_f3_draft_candidates=0`;
- `fact_gated_manager_only_exact_proof_needs_policy=2`;
- `fact_gated_already_self_exact_proof=6`;
- `fact_gated_blocked_no_exact_proof=1`;
- `fact_gated_excluded_danger_money_p0=1`.

## Вывод

F2u подтверждает регрейд Claude #1:

- быстрый route-only active по ack/status сейчас не имеет рычага;
- активировать Ф3 нельзя: strict draft-кандидатов `0`;
- реальная полезная зона - proof/retrieval/policy для stable existence/format;
- две строки с exact KB proof остаются `manager_only`, то есть требуют отдельного
  policy/upstream решения, а не простого route demotion.

## Инварианты

- runtime/direct path/provider/profile/live не изменены;
- route/text не меняются;
- P0-floor/preblock не тронут;
- AMO/Tallanto/CRM/Telegram не тронуты;
- полный клиентский текст/`client_safe_text`/`template_text` в новый summary не
  экспортируется.
