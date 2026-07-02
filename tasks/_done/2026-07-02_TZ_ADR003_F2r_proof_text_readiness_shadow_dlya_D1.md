Ветка: codex/adr003-semanticframe-migration
Зоны: scripts/report_adr003_frame_calibration_queue.py, tests/test_report_adr003_frame_calibration_queue.py, tasks/_done/2026-07-02_TZ_ADR003_F2r_proof_text_readiness_shadow_dlya_D1.md
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_report_adr003_frame_calibration_queue.py
Семантический-аудит: да

# ADR-003 F2r: Proof Reconciliation Text-Readiness Shadow

## Контекст

F2q показал полезный shadow-сигнал: свежий точный факт иногда противоречит `missing_facts` в SemanticFrame. Но это не доказывает, что текущий текст уже можно отдавать клиенту без менеджера.

## Задача

Расширить report-only очередь калибровки диагностикой `text_readiness` для `semantic_frame_proof_reconciliation_shadow`.

Разрешено:

- читать только существующую telemetry из транскриптов;
- считать причины блокировки текста;
- писать только отчётные поля.

Запрещено:

- менять runtime route/text;
- включать новый модельный вызов;
- добавлять regex-понимание смысла;
- менять `provider.py`, direct path, профиль, P0-floor/preblock;
- делать active-вердикт.

## Приёмка

- Все новые строки остаются `active_behavior_allowed=false`.
- Отчёт показывает, сколько proof-reconciliation строк:
  - уже без route-рычага;
  - заблокированы `manager_only`;
  - заблокированы deferral/missing_facts/output-gates/verifier-unavailable;
  - являются только manual-review кандидатами на send-as-is.
- На текущем F2q-сырье active остаётся NO-GO.
- Тесты зелёные.
