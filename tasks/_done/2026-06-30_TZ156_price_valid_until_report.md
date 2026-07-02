# ТЗ-156 — отчёт D6

Ветка: `codex/tz156-price-valid-until`
База: `main@706c0d9`

## Что сделано

- 12 ценовых фактов из ТЗ продлены до `valid_until=2026-12-31`.
- Числа цен, бренды, `fact_key` и `allowed_for_client_answer=True` не менялись.
- В YAML-источнике добавлен явный `valid_until` через leaf-формат `raw_value + valid_until`, чтобы пересборка не возвращала дату из `before_2026_07_01` / `before_2026_08_01`.
- Пиннутый релиз `kb_release_20260612_v6_7_staging_r4_1` пересобран через `scripts/build_kb_release_v6_1_team_answers.py`.
- Сохранены прежние `venue/program_kind` метки, чтобы не менять поведение venue-scope вне задачи.

## Проверки

- Пересборка KB: `quality_passed=true`, `semantic_pass=true`, `semantic_blocking_findings=0`.
- Целевая сверка: `target_count=12`, `price_diffs=[]`, `deadline_mentions=[]`, `bad_valid_brand_allowed=[]`.
- Точечные тесты KB/direct-path: `357 passed`.
- Полный pytest: `3753 passed, 5 skipped, 1 warning`.

## Audit pack

`audits/_inbox/tz156_price_valid_until_20260630_20260630193836`

## Границы

Живой бот, CRM, AMO, Tallanto и отправки клиентам не запускались.
