Ветка: codex/adr003-semanticframe-migration
Зоны: scripts/report_adr003_frame_calibration_queue.py, tests/test_report_adr003_frame_calibration_queue.py, audits/_inbox/adr003_f2i_frame_calibration_queue_20260702064000/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_report_adr003_frame_calibration_queue.py tests/test_report_adr003_overhandoff_levers.py tests/test_report_adr003_fact_gated_self_answer_readiness.py tests/test_report_adr003_exact_proof_injection_shadow.py
Семантический-аудит: да

# ADR-003 F2i: SemanticFrame Calibration Queue

## Контекст

После F2h активное понижение route остаётся NO-GO: свежий exact proof сам по себе не снимает блокеры `manager_only`, `context_update`, runtime missing live facts и ошибки самого SemanticFrame.

Комментарий Claude #1 уточнил реальный рычаг: не price и не harmless ack/status, а путаница между:

- безопасной справкой “существует ли курс/формат/класс”;
- live availability / запись / проверка мест / действие менеджера.

## Задача

Сделать report-only инструмент, который строит очередь калибровки SemanticFrame и соседних слоёв по реальному M1-прогону `36ea110`.

Обязательные границы:

- не менять route/text;
- не менять runtime, prompt, direct path, profile или P0-floor;
- не делать новых модельных вызовов;
- не трогать live Telegram/Wappi/AMO/CRM/Tallanto;
- все строки отчёта должны быть `active_allowed=false`.

## Приёмка

- Отчёт разделяет manual `frame_too_cautious` label и настоящую ошибку поля `frame.must_handoff`.
- Отчёт показывает workstreams:
  - `semanticframe_existence_vs_availability`;
  - `semanticframe_safe_reference_missing_facts`;
  - `semanticframe_low_confidence`;
  - `retrieval_delivery_runtime_missing_exact_proof`;
  - `conversation_plan_scope_missing`;
  - `policy_manager_only_exact_proof`;
  - `policy_context_update_exact_proof`;
  - `danger_adjacent_do_not_lower`;
  - `already_self_no_active_leverage`;
  - `measurement_review_unclear`.
- На реальном прогоне 36ea110 отчёт даёт `active_readiness=no_go` и `strict_active_candidates_now=0`.
- Тесты зелёные.
- Audit pack содержит semantic review.

## Результат

Сделано:

- `scripts/report_adr003_frame_calibration_queue.py`;
- `tests/test_report_adr003_frame_calibration_queue.py`;
- audit pack `audits/_inbox/adr003_f2i_frame_calibration_queue_20260702064000/`.

Реальный прогон 36ea110:

- safe/self gold rows: 32;
- manual too-cautious labels: 29;
- true frame must_handoff too-cautious: 14;
- true frame too-confident: 0;
- current safe over-handoff: 11;
- strict active candidates now: 0;
- manager-only exact-proof rows: 2.

Вердикт: F2i report-only PASS, active NO-GO.
