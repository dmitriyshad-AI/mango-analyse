# D7 CRM card readiness text-quality gate, v5 rawdirect preview

Дата: 2026-06-24

## Итог

- Закрыт readiness bug: если `crm_text_quality_detector` находит P1/P2/P0 finding, карточка получает `Готово=нет`.
- Причина переносится в `Блокеры` человекочитаемо: `Текст карточки требует проверки (...)`.
- Fallback для `mango_call.summary/text_preview` из предыдущего блока сохранён.
- Bot-safe путь не менялся.
- AMO/Tallanto/CRM write: 0.
- Вердикт на live/production не выносился.

## Изменения

Файлы:

- `/Users/dmitrijfabarisov/Projects/Mango_etap1_crm_cards/src/mango_mvp/crm_card_aggregator.py`
- `/Users/dmitrijfabarisov/Projects/Mango_etap1_crm_cards/tests/test_crm_card_aggregator.py`

Что сделано:

- Добавлен `_crm_text_quality_blockers(contact_payload, deal_payload)`.
- Для проверки используется тот же `detect_crm_text_quality_risks(..., min_severity="P2")`, что и writeback quality gate.
- Payload мапится из preview-полей в реальные AI-поля:
  - `История общения` -> `Авто история общения`;
  - `Последняя сводка` -> `Последняя AI-сводка`;
  - `Статус сделки` -> `AI-фактический статус сделки`;
  - `Следующий шаг` -> `AI-рекомендованный следующий шаг`;
  - `Tallanto` -> `AI-Tallanto статус по сделке`;
  - `Предупреждения` -> `AI-предупреждение по сделке`.
- Blocking finding добавляется в contact/deal blockers до расчёта `ready`.
- Добавлен NEG-тест: wrong-person text-quality finding переводит `workbook.ready`, `contact_card.ready_for_amo`, `deal_card.ready_for_amo` в `нет/False`.

## Preview v5

Команда:

```bash
DB="/Users/dmitrijfabarisov/Projects/Mango_botsafe_summary_builder/.codex_local/botsafe_summary_builder_20260624/customer_timeline_wide_classfix_v5_rawdirect.sqlite"
OUT="/Users/dmitrijfabarisov/Projects/Mango_etap1_crm_cards/product_data/customer_timeline/crm_card_preview_20260624_d4_wide_classfix_v5_rawdirect_readiness"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/build_crm_customer_card_workbook.py \
  --timeline-db "$DB" \
  --allowed-root "/Users/dmitrijfabarisov/Projects/Mango_botsafe_summary_builder/.codex_local/botsafe_summary_builder_20260624" \
  --out-xlsx "$OUT/crm_cards_preview.xlsx" \
  --tenant-id foton \
  --sample-size 200
```

Артефакты:

- `/Users/dmitrijfabarisov/Projects/Mango_etap1_crm_cards/product_data/customer_timeline/crm_card_preview_20260624_d4_wide_classfix_v5_rawdirect_readiness/crm_cards_preview.xlsx`
- `/Users/dmitrijfabarisov/Projects/Mango_etap1_crm_cards/product_data/customer_timeline/crm_card_preview_20260624_d4_wide_classfix_v5_rawdirect_readiness/crm_cards_preview.csv`
- `/Users/dmitrijfabarisov/Projects/Mango_etap1_crm_cards/product_data/customer_timeline/crm_card_preview_20260624_d4_wide_classfix_v5_rawdirect_readiness/crm_cards_preview.summary.json`
- `/Users/dmitrijfabarisov/Projects/Mango_etap1_crm_cards/product_data/customer_timeline/crm_card_preview_20260624_d4_wide_classfix_v5_rawdirect_readiness/direct_preview_quality_summary.json`
- `/Users/dmitrijfabarisov/Projects/Mango_etap1_crm_cards/product_data/customer_timeline/crm_card_preview_20260624_d4_wide_classfix_v5_rawdirect_readiness/examples_10.json`

## Метрики

| Метрика | Значение |
|---|---:|
| Строк | 200 |
| `ready_yes` | 60 |
| `ready_no` | 140 |
| `ready_yes` с text-quality blocking | 0 |
| Text-quality blocking rows всего | 19 |
| Text-quality warning rows | 86 |
| Бренд-смешение в `ready_yes` | 0 |
| Бренд-смешение во всём preview | 2 |
| Raw JSON/service-id/мусор | 0 |
| История общения непустая | 184 |
| Последняя сводка непустая | 184 |

Workbook blockers:

| Блокер | Кол-во |
|---|---:|
| На телефоне несколько человек — проверьте, к кому относится | 91 |
| Не найдена сделка в AMO | 74 |
| Не найден контакт в AMO | 38 |
| Текст карточки требует проверки: история общения пустая | 16 |
| Текст карточки требует проверки: оплата/чек уже упомянуты, следующий шаг может противоречить | 1 |
| Текст карточки требует проверки: возможная путаница с клиентом или неверный контакт | 1 |
| Текст карточки требует проверки: следующий шаг противоречит закрытию или пассивному отказу | 1 |

Text-quality классы:

| Risk | Кол-во |
|---|---:|
| `empty_auto_history` | 16 |
| `completed_payment_next_step_conflict` | 1 |
| `wrong_person_or_identity_mismatch` | 1 |
| `closure_next_step_requires_downgrade` | 1 |

## 10 примеров

| Row | customer_id | brand | Готово | Блокеры |
|---:|---|---|---|---|
| 1 | `customer:0056897822baaaf456fc9a7b8219d9f8` | `unpk` | да | - |
| 2 | `customer:0152bccef9dcb4c1a4df15308f58864b` | `unknown` | да | - |
| 3 | `customer:05f9d5e0bc65b858bafef2304cc91e28` | `unknown` | да | - |
| 4 | `customer:0604b62ae62127df016bda1291e6b59a` | `unknown` | да | - |
| 5 | `customer:069838b4de11441cb563fe705b97eb7e` | `unknown` | да | - |
| 6 | `customer:08bcb6c436d4ab9fb70e5f0b3bac4d00` | `unknown` | нет | На телефоне несколько человек — проверьте, к кому относится |
| 7 | `customer:099027984a147d927cbfd76dd2a831d7` | `unknown` | да | - |
| 8 | `customer:0b43ed43be5e40a0cdfedfb7f981a3db` | `unpk` | да | - |
| 9 | `customer:0be87264e87b17708c33b179caf6c855` | `unpk` | нет | ambiguous phone + text-quality payment/next-step conflict |
| 10 | `customer:0c09eeb7bcf29119916efe532468c39e` | `unknown` | да | - |

## Тесты

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_crm_card_aggregator.py
# 12 passed in 0.53s

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q
# 3382 passed, 5 skipped, 1 warning in 50.67s
```

## Semantic review

Verdict: `PASS_WITH_NOTES`.

Что прошло:

- Строки с P1/P2 text-quality finding больше не остаются `ready_yes`.
- Причина блокировки видна человеку в `Блокеры`.
- Полезный manager-only fallback сохранён, но bot-safe не расширен.
- В `ready_yes` нет бренд-смешения и нет raw JSON/service-id мусора.
- AMO/Tallanto/CRM write не выполнялся.

Notes:

- Во всём preview остаются 2 cross-brand строки, обе `ready=нет`; это не мешает будущему dry-run, но для UX можно отдельной правкой чистить/маскировать cross-brand fragments в заблокированных строках.
- `Предупреждения` как отдельное поле пока не заполняется text-quality причинами; причина уже есть в `Блокеры`, что достаточно для readiness.
- Вердикт на live/write не давался.
