> DONE 2026-07-02 04:13 | ветка codex/adr003-semanticframe-migration | codex

> TAKE 2026-07-02 04:08 | ветка codex/adr003-semanticframe-migration | codex

Ветка: codex/adr003-semanticframe-migration
Зоны: scripts/report_adr003_existence_fact_verification.py, src/mango_mvp/knowledge_base/product_existence_axes_catalog.py, tests/test_report_adr003_existence_fact_verification.py, tests/test_kb_product_existence_axes_catalog.py, tasks/, audits/_inbox/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_report_adr003_existence_fact_verification.py tests/test_kb_product_existence_axes_catalog.py
Семантический-аудит: да

# TZ ADR-003 F2e: F2c report uses product existence catalog

## Контекст

F2d добавил строгий `product_existence_axes_catalog`: `exists | not_offered | unknown | needs_slot`.

Старый F2c report (`report_adr003_existence_fact_verification.py`) всё ещё использует временные строковые совпадения по KB (`_axis_hits`, `_kb_matches`). Это годится как diagnostic, но не должно быть следующим proof-контрактом.

## Цель

Перевести F2c report на новый product-existence catalog:

- `build_product_existence_axes_catalog(kb_facts)`;
- `verify_product_format_exists(...)`;
- `exists`/`not_offered` считаются exact KB evidence;
- `unknown`/`needs_slot` не считаются exact evidence.

## Scope

- Только `scripts/report_adr003_existence_fact_verification.py`.
- Только тесты отчёта.
- Никакой runtime-проводки, direct path, provider, profile, P0 floor/preblock.

## Инварианты

- Поведение бота не меняется.
- Отсутствие факта не становится `not_offered`.
- Payment/enrollment/manager-action факты не становятся proof.
- Wrong brand не матчится.
- Danger/P0/money rows остаются excluded.
- Report должен явно писать `product_existence_check` для сырьевой сверки Claude #1.

## Acceptance

- Тесты отчёта + тесты каталога зелёные.
- Audit pack показывает, что F2c теперь использует F2d proof-layer.
- `git diff --check` чистый.

## Допуск по F2d

Если F2e на реальном отчёте показывает, что proof-layer не понимает обычную форму класса вроде `5-й класс`, разрешено сузко поправить нормализацию grade в `product_existence_axes_catalog.py` с регрессионным тестом. Это остаётся частью proof-layer, не runtime-проводкой.
