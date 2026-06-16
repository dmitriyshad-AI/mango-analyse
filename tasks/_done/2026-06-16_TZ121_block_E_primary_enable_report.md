# TZ-121 Block E Primary Enable

Дата: 2026-06-16  
Ветка: `codex/tz121-group4-remaining`

## Что включено

- `cyrillic_v2` включен как основной офлайн-резолвер бренда для `canonical_readonly_import` и `canonical_readonly_triage`.
- Старый режим `infer_brand(..., mode="legacy")` сохранен для совместимости и NEG-проверок.
- Добавлен полевой резолвер `infer_offline_brand`: явный Foton в истории не блокируется одиночными техническими/историческими `МФТИ`-следами, но явное смешение брендов остается `unknown`.
- Небрендовые формы `фотончики` и `Фотоний` не считаются брендом Foton.

## Follow-up перед primary

Источник ручного gold-среза:

`/Users/dmitrijfabarisov/Projects/Mango_tz116_offline/audits/_inbox/tz116_followup_gold_reviews_20260615/e_brand_loss_gold_sample.csv`

Источник реальных строк read-only:

`/Users/dmitrijfabarisov/Projects/Mango analyse/stable_runtime/sales_master_export_20260523_audio_working_store_v1/master_contacts_ru.csv`

Артефакт:

`audits/_inbox/tz121_e_brand_followup_real_20260616/`

Счетчики:

- строк всего: `20`;
- `foton->unknown` false-negative: `9`;
- исправлено в `foton`: `9/9`;
- expected fail-closed: `8`;
- осталось `unknown`: `8/8`;
- unclear: `2`, не используются как разрешение на primary;
- `llm_calls_total`: `0`;
- записи в AMO/Tallanto/CRM/DB: `0`.

Примечание: один `unpk->unknown` false-negative из старого follow-up остается не исправленным этим блоком; E-primary по текущей задаче закрывает Foton-морфологию и не ослабляет cross-brand.

## Измененные файлы

- `src/mango_mvp/customer_timeline/canonical_readonly_import.py`
- `src/mango_mvp/customer_timeline/canonical_readonly_triage.py`
- `scripts/run_tz121_brand_e_followup_real.py`
- `tests/test_customer_timeline_canonical_readonly_import.py`
- `tests/test_tz121_brand_e_followup_real.py`

## Проверки

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q \
  tests/test_customer_timeline_canonical_readonly_import.py::test_infer_brand_cyrillic_v2_foton_root_and_cross_brand_fail_closed \
  tests/test_tz121_brand_e.py \
  tests/test_tz121_brand_e_followup_real.py

4 passed
```

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/run_tz121_brand_e_followup_real.py \
  --out-dir audits/_inbox/tz121_e_brand_followup_real_20260616

gate_passed=true
```

## Статус

`formal_pass`: да.  
`semantic_pass`: см. `tasks/_done/2026-06-16_TZ121_block_E_primary_semantic_review.md`.  
Следующий блок по ТЗ-121: C hybrid shadow на микро-наборе, затем стоп на регрейд.
