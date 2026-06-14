# TZ-102 direct path model P0 report

Дата: 2026-06-14
Ветка: `codex/tz102-model-p0-direct`
База: `codex/tz26-action-decision`

## Что сделано

- Добавлен флаг `TELEGRAM_DIRECT_PATH_MODEL_P0`, default OFF.
- Флаг не добавлен в `pilot_gold_v1`.
- В direct path prompt под флагом добавлены:
  - инструкция по срочным обращениям: спорная оплата, списание/платёж, возврат, жалоба, юридическая угроза;
  - поля JSON `is_p0`, `risk_level`, `p0_kind`, `model_reason`;
  - разрешение route `manager_only` только для срочного/P0.
- `_normalize_direct_path_payload` сохраняет структурный модельный сигнал в `metadata["direct_path_model_p0"]`.
- В `_build_direct_path_draft` добавлен первичный guard до `apply_semantic_output_verifier` и `apply_authoritative_output_gate`:
  - источник = `model_p0 ∪ p0_pre_gate`;
  - модель может только добавить P0, не снять;
  - route переводится в `manager_only` до финального гейта;
  - в `safety_flags` пишется latchable код (`payment_dispute/refund/complaint/legal_threat`) и точный флаг `direct_path_model_p0_<kind>`.
- `answer_safety_classifier` читает только точные новые флаги `direct_path_model_p0_*`, чтобы authoritative gate сам сформировал `hard_p0` и заменил продающий текст.
- `_deal_action_final_p0` учитывает `metadata["direct_path_model_p0"]`, чтобы слой действий не предлагал продающий шаг.

## Паритет контуров

- Pipeline ADR-002 не менялся: существующий путь `model_p0 OR p0_pre_gate` до маршрута оставлен как есть.
- Direct path с флагом OFF сохраняет старое поведение; это покрыто NEG.

## Проверки

- Новые NEG:
  - флаг OFF не меняет маршрут/текст даже при модельном `is_p0=true`;
  - real_006-класс: «спорная ситуация с оплатой» → `manager_only`, продающий текст заменён гейтом, `hard_p0`;
  - модельный `payment_dispute` попадает в `p0_latch.codes`, следующий нейтральный ход preblock-ится как P0;
  - benign-набор «дорого/подумаю» и гипотетический возврат остаются автономными;
  - prompt под флагом содержит `is_p0/risk_level/p0_kind/model_reason`, без флага не содержит.
- Зона direct/action/classifier/memory:
  - `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_subscription_llm_draft_provider.py tests/test_deal_action_decision.py tests/test_answer_safety_classifier.py tests/test_dialogue_memory.py`
  - Результат: `595 passed`.
- Полный pytest:
  - `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests`
  - Результат: `3209 passed, 2 skipped, 1 warning`.

## Semantic review

Вердикт: `PASS_WITH_NOTES`.

Что проверено:
- клиентский продающий текст на модельном P0 не выходит наружу;
- route становится `manager_only` до финального гейта;
- latch удерживает следующий нейтральный ход;
- benign-примеры не получают P0 при `is_p0=false`.

Остаток:
- живой LLM/сырьё real_006 не гонялись в этом блоке; ревью по сырью остаётся за Claude #1.
- Числовой порог уверенности не зашит, как требует ТЗ; нужна будущая калибровка на золотом наборе ADR-002.
