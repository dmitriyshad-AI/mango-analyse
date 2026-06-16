# ТЗ-126: честная метрика over-handoff

Дата: 2026-06-16  
Ветка: `codex/tz126-overhandoff-metric`

## Что сделано

- В `scripts/run_telegram_dynamic_client_sim.py` добавлен post-processing классификатор `_classify_handoff_bucket`.
- Классификация применяется только в summary-метрике over-handoff; поведение бота и тексты ответов не меняются.
- В `summary["over_handoff"]["buckets"]` добавлены:
  - `counts`: `closing`, `legitimate`, `disputed_p0`, `upsell_miss`, `unclassified`;
  - `shares`: доли внутри handoff-turns;
  - `examples`: до 10 примеров на корзину.
- P0-уходы теперь попадают в denominator метрики и раскладываются как `legitimate` или `disputed_p0`; иначе P0-ветка из ТЗ была бы недостижима, потому что прежний `_is_over_handoff_turn` полностью отбрасывал P0.

## Инварианты

- Живой путь бота не тронут: изменения только в `scripts/run_telegram_dynamic_client_sim.py` и тестах.
- `channels/*`, `provider`, `post_layers`, CRM/Tallanto/AMO, ASR и runtime DB не менялись.
- Классификатор переиспользует сохранённый `bot_close_detect.status`; новый детектор закрытия/благодарности не добавлялся.
- `wrong_scope` считается `upsell_miss`.
- Закрытие с вопросом не попадает в `closing`.
- `contact_requested`/`capture_lead` без открытого вопроса уходит в `legitimate`.

## Проверки

- Точечный прогон: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_telegram_dynamic_client_sim.py`
  - Результат: `105 passed`.
- Полный прогон: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q`
  - Результат: `3308 passed, 5 skipped, 1 warning`.

## Observe-only проверка

- Юнит-тестом подтверждено, что `build_summary` не мутирует `transcripts`.
- Read-only пересчёт доступного C0 TZ123:
  - Источник: `/Users/dmitrijfabarisov/Projects/Mango_tz113_114_115_profile/runs/20260616_tz123_c0_surface_real_OFF/dynamic_dialog_transcripts.jsonl`
  - SHA до/после: `441ebb4b36474b11aa72b7072a4fd36b938b1d1ba93afb1452bed54572413eef`
  - `byte_identical=True`
  - `dialogs=3`, `handoff_turns=0`
- Read-only пересчёт старого over-handoff набора:
  - Источник: `/Users/dmitrijfabarisov/Projects/Mango analyse/audits/_inbox/over_handoff_wave1_tripwire_20260527_043422/dynamic_dialog_transcripts.jsonl`
  - SHA: `22d33985ae762f8e00843d703943c13c172735a41fecab29c5472228c8bb02d4`
  - `byte_identical=True`
  - `dialogs=8`, `handoff_turns=21`
  - buckets: `closing=0`, `legitimate=10`, `disputed_p0=0`, `upsell_miss=11`, `unclassified=0`

## Ограничение C0-валидации

Полную C0-валидацию из ТЗ по ожидаемым долям (`closing≈39%`, `legitimate≈10%`, `upsell_miss≈15-20%`, FN-дожима=0) я не считаю закрытой: доступные C0/старые transcript-файлы либо имеют 0 handoff-turns, либо не содержат `bot_close_detect`, поэтому корзина `closing` на них неизбежно нулевая. Код готов для регрейда на правильном C0-сырье с `bot_close_detect`.

## Файлы

- `scripts/run_telegram_dynamic_client_sim.py`
- `tests/test_telegram_dynamic_client_sim.py`
- `tasks/_done/2026-06-16_TZ126_overhandoff_metric_report.md`

## llm_calls_total

`0` — модель не вызывалась.
