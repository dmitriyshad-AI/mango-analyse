# D7 CRM card service snapshot cleanup

Дата: 2026-06-20

## Что исправлено

- `amo_contact_snapshot`, `tallanto_student_snapshot`, `amo_deal_stage` больше не попадают в `Авто история общения`, `Хронология`, `Последняя AI-сводка` и содержательную историю сделки.
- `Последняя AI-сводка` теперь берётся из последнего содержательного звонка `mango_call` с `call_history_eligible=True`; если звонка нет, используется безопасный fallback по старым фактам/неслужебной истории.
- `AI-рекомендованный следующий шаг` берётся из `call_analysis.next_step`, если он есть.
- `AI-актуальные возражения` берутся из `call_analysis.objections`; агрегатная строка вида `Клиент в истории с...` больше не подставляется под метку `Возражения`.
- `interests` и `target_product` из `call_analysis` выводятся в автоисторию как `Интересы` и `Целевой продукт`.
- `amo_deal_stage` вынесен в `AI-фактический статус сделки`; из `AI-сводка по сделке` статус убран, чтобы не смешивать статус и содержательную сводку.
- Сырые Tallanto-маркеры `exact_phone_single` и `no_exact_phone_match` переводятся в человекочитаемый текст.

## Preview

Путь:

`/Users/dmitrijfabarisov/Projects/Mango_etap1_crm_cards/product_data/customer_timeline/crm_card_preview_20260620_full_call_analysis/crm_cards_preview.xlsx`

Сводка на 200 клиентах Фотона:

- `ready_yes`: 68
- `ready_no`: 132
- `rows_with_next_step`: 145
- `rows_with_objections`: 157

Контроль по `ЧТО ПОЙДЁТ В AMO`:

- `Read-only AMO contact snapshot`: 0
- `exact_phone_single`: 0
- `no_exact_phone_match`: 0
- плохих `Последняя AI-сводка` со служебным статусом/маркерами: 0

## Тесты

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_crm_card_aggregator.py` -> `7 passed`
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests` -> `3377 passed, 5 skipped, 1 warning`

Предупреждение прежнее внешнее: `urllib3 NotOpenSSLWarning`.

## Safety

- AMO/Tallanto live-write не запускался.
- Timeline/source DB читались read-only.
- Preview CSV/JSON/XLSX оставлены локальными артефактами и не добавлены в git.
