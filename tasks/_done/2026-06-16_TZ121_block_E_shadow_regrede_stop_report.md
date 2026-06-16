# TZ-121 Блок E: бренд, shadow на микро-наборе

Дата: 2026-06-16

## Статус

Блок E выполнен до точки остановки на регрейд.

E primary не включался. Блок C не начинался.

## Что реализовано

- В `infer_brand` добавлен отдельный режим `mode="cyrillic_v2"`.
- Default остаётся `legacy`, то есть старое поведение не меняется без явного режима.
- `cyrillic_v2` нормализует регистр и `ё -> е`.
- Корень `фотон` ловит склонения и склейки:
  - `Фотона`
  - `Фотоны`
  - `Фотону`
  - `в Фотоне`
  - `ЦДПФОТОН`
  - `ЦИДПОФОТОН`
- Корни УНПК:
  - `унпк`
  - `мфти`
- Если найдены оба бренда, результат `unknown` по fail-closed.

## Shadow-замер

Артефакты для регрейда:

- `audits/_inbox/tz121_e_brand_micro_shadow_20260616/summary.json`
- `audits/_inbox/tz121_e_brand_micro_shadow_20260616/tz121_e_brand_trace.csv`
- `audits/_inbox/tz121_e_brand_micro_shadow_20260616/tz121_e_brand_trace.jsonl`
- `audits/_inbox/tz121_e_brand_micro_shadow_20260616/REPORT.md`

Счётчики:

- строк: `12`
- model_correct: `12`
- model_break: `0`
- Foton gold rows: `7`
- Foton unknown legacy: `0`
- Foton unknown cyrillic_v2: `0`
- Cross-brand rows: `2`
- Cross-brand fail-closed: `2`
- error types: `{"both_correct": 9, "model_fix": 3}`
- `llm_calls_total=0`

Примечание: в текущем `main` legacy уже ловил большинство форм Фотона через подстроку `фотон`, поэтому на микро-наборе нет снижения `foton->unknown` с ненулевого значения. Ценность E в этом срезе: закрепить корневой режим, добавить `МФТИ -> unpk` и закрыть cross-brand в `unknown`.

## Безопасность

- Только синтетический микро-набор.
- Полные наборы не запускались.
- Модель не вызывалась.
- AMO/Tallanto/CRM не трогались.
- БД и `stable_runtime` не читались и не писались.
- E primary не включался.

## Проверки

Точечно:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_customer_timeline_canonical_readonly_import.py::test_infer_brand_cyrillic_v2_foton_root_and_cross_brand_fail_closed tests/test_tz121_brand_e.py tests/test_outcome_linker.py tests/test_tz121_outcome_b.py
```

Результат: `19 passed`.

Полный pytest:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q
```

Результат: `3292 passed, 5 skipped, 1 warning`.

Shadow:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/run_tz121_brand_e_micro_shadow.py --out-dir audits/_inbox/tz121_e_brand_micro_shadow_20260616
```

Результат: см. счётчики выше.

## Стоп

Остановиться на регрейд Claude/Dmitry. До регрейда:

- не включать E primary;
- не переходить к C;
- не запускать полные сеты.
