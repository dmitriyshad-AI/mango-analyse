> DONE 2026-06-30 15:30 | ветка main | codex

> TAKE 2026-06-30 15:04 | ветка main | codex

Ветка: main
Зоны: src/mango_mvp/channels/subscription_llm_parts/, product_data/telegram_dynamic_test_sets/, scripts/, tests/, tasks/, docs/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_subscription_llm_draft_provider.py
Семантический-аудит: да

# ТЗ-153 — ADR-003 Этап 0+1: заморозка eval-набора + SemanticFrame в SHADOW (без смены поведения)

- Дата: 2026-06-30. Постановщик: Claude #1 (+аудитор, +код-сверка D1). Исполнитель: главный Codex D1. Регрейд — Claude #1 по сырью. Канон: main (eb6fa0b или новее — сверь HEAD).
- Основание: ADR-003 v2 (`Foton/2026-06-30_ADR-003_...md`), валидирован тройной независимой сверкой. Это shadow-этап: поведение бота НЕ меняется. Ноль изменений route/text.
- Дисциплина: за флагом default OFF; LLM только через подписку Codex (без ключей); frame НЕ отдельный модельный вызов — встраивается в существующий direct-path payload; P0-пол не трогаем; регрейд по сырью.

## Этап 0 — заморозить eval-набор + мораторий

1. Собрать версионированный eval-набор реальных провалов: 25 Wappi what-if (из `~/.mango_local/draft_loop_whatif/pair_missing_72h_latest25_*`) + существующие наборы P0 / brand / product-регрессий. Зафиксировать как именованный набор (с версией/датой), воспроизводимо гоняемый на M1.
2. Снять baseline-метрики на нём: send-as-is по ходам, P0 fail-closed pass, brand 0 утечек, fabrication hard-gate 0. Числа должны быть стабильны между 2 прогонами.
3. Мораторий: новый провал -> кейс в eval-набор, НЕ новый детектор/SAFE_TEXT/флаг понимания.

## Этап 1 — SemanticFrame в SHADOW (телеметрия, без влияния)

1. Встроить frame в существующий direct-path payload (не новый вызов). Модель уже отдаёт P0/intent/action/answerability в одном draft-payload (`direct_path.py:2500`); расширить выходную схему полями frame: `{intent, risk_class, deal_stage, payment_readiness, requested_product, requested_action, answerability, must_handoff, evidence, confidence}`. `deal_stage`/`payment_readiness` сейчас отсутствуют — их производит frame.
2. Писать frame в метаданные/телеметрию, route/text НЕ менять. Прецедент — `answerability_trace` (`post_layers.py:1931`): пишется без смены решения. Флаг `TELEGRAM_SEMANTIC_FRAME_SHADOW` default OFF.
3. Отдельно измерить на eval-наборе (frame vs текущие детекторы): P0-recall (`frame.must_handoff` vs фактический P0-исход), false positives, relevance-сигнал, product-existence-сигнал. Плюс отчёт расхождений frame↔детекторы (intent-plan/tone_close/deal-action).

## Приёмка

- route/text бит-в-бит не меняются на всём eval-наборе (diff финальных черновиков = 0).
- +0 модельных вызовов (frame консолидирован в существующий draft-payload; подтвердить по числу вызовов на ход).
- `frame.must_handoff` совпадает с фактическим P0-исходом >=95% на P0-наборе.
- Отчёт расхождений frame↔детекторы на 25 Wappi + регрессиях собран.
- Полный pytest зелёный. P0-пол не тронут.

## Явно НЕ в этом ТЗ

- Возврат complaint в P0-union.
- Разведение payment/refund/dispute.
- product-existence gate.
- tone_close -> модель.
- DecisionPolicy, право менять route/action, relevance-инвариант, фикс порядка слоёв.

## Файлы

- `direct_path.py:2500` — расширить payload-схему.
- `provider.py:2157+` — распарсить frame в метаданные.
- `post_layers.py:1931` — прецедент shadow-trace.
- Точка записи телеметрии в draft_loop/journal.
- Eval-набор — `product_data/telegram_dynamic_test_sets/` + Wappi what-if.
