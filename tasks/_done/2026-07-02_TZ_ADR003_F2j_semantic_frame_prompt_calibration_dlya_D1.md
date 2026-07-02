Ветка: codex/adr003-semanticframe-migration
Зоны: src/mango_mvp/channels/subscription_llm_parts/direct_path.py, src/mango_mvp/channels/subscription_llm_parts/provider.py, tests/test_direct_path_semantic_frame_shadow.py, audits/_inbox/adr003_f2j_semantic_frame_prompt_calibration_20260702071500/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_direct_path_semantic_frame_shadow.py tests/test_report_adr003_frame_calibration_queue.py tests/test_report_adr003_frame_gold_calibration.py tests/test_report_adr003_semantic_frame_eval.py
Семантический-аудит: да

# ADR-003 F2j: SemanticFrame Prompt Calibration

## Контекст

F2i показал реальный класс ошибки: SemanticFrame путает безопасную справку “есть ли курс/лагерь/формат для класса X” с live availability / записью / проверкой мест.

## Задача

Откалибровать только SemanticFrame shadow prompt:

- inline prompt в `direct_path.py`;
- posthoc prompt в `provider.py`.

Не менять:

- route/text;
- P0 floor;
- profile flags;
- direct-path runtime routing;
- manager-only policy;
- live Telegram/Wappi/AMO/CRM/Tallanto.

## Приёмка

- Prompt явно говорит: stable existence/format reference = `answer_question`, не `check_availability`.
- Prompt явно говорит: seats/booking/enrollment/live group = `check_availability/manager_only`.
- Тесты фиксируют эту границу для inline и posthoc prompt.
- Локальный posthoc measurement на 7 проблемных F2i dialogs показывает сдвиг по `requested_action`.
- Active остаётся NO-GO.

## Результат

Локальный posthoc measurement на subset:

- `requested_action_wrong`: 6 -> 1;
- `check_availability` исчез из subset;
- `must_handoff_wrong`: 8 -> 8;
- `risk_class_wrong`: 9 -> 9;
- `answerability_wrong`: 9 -> 9.

Вердикт: полезная частичная калибровка action field, но active NO-GO. Следующий блокер — `risk_class=missing_facts` / `answerability=manager_only` на safe reference.
