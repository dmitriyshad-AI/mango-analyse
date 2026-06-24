# D7 CRM card fallback preview on D4 wide classfix v4

Дата: 2026-06-24

## Короткий итог

- В `crm_card_aggregator` добавлен manager-only fallback: если у `mango_call` нет `call_analysis` и нет `call_history_eligible`, но есть содержательный `summary/text_preview`, он попадает в историю карточки менеджера.
- Fallback не используется для bot-safe: в NEG-тесте `bot_safe_fields == []`.
- `technical/non_conversation` отсекаются по `call_type` и по текстовым маркерам недозвона/технического звонка.
- Прямой preview выполнен на исправленной D4-копии, без `manager_facts_csv`.
- AMO/Tallanto/CRM write: 0. Safety summary: `write_amo=false`, `write_tallanto=false`, `write_customer_timeline=false`, `live_network_calls=false`.
- Готовность preview: `ready_yes=62`, `ready_no=138` из 200.
- История/сводка появились в 184/200 строках.
- Raw JSON/service-id/технический мусор в видимых полях карточки: 0.
- Бренд-смешение в строках, готовых к записи: 0. В полном preview осталось 2 cross-brand попадания, оба в `ready=нет` строках.
- Text-quality P1/P2: 19 строк всего; среди `ready=да` 2 строки, обе выглядят как полезные стоп-кейсы, а не как полезные записи.
- Вердикт на использование/live не выносился.

## Изменения

Файлы:

- `/Users/dmitrijfabarisov/Projects/Mango_etap1_crm_cards/src/mango_mvp/crm_card_aggregator.py`
- `/Users/dmitrijfabarisov/Projects/Mango_etap1_crm_cards/tests/test_crm_card_aggregator.py`

Суть:

- `_history_events()` и `_latest_history_call()` теперь используют `_mango_call_manager_history_eligible()`.
- Новый fallback включается только когда `call_analysis` и `call_history_eligible` отсутствуют.
- Если `call_history_eligible` явно есть, прежнее поведение сохраняется.
- Если `call_analysis` есть, даже пустой, fallback не включается.
- `call_type=non_conversation/technical` и текстовые признаки недозвона не попадают в историю.

## NEG

- Полезный `mango_call.summary` без `call_analysis/call_history_eligible` попадает в `Последняя сводка`.
- Технический/неразговорный звонок рядом не попадает.
- Fallback не создаёт `Следующий шаг`, потому что это только история менеджера, не извлекатель действий.
- Bot-safe поля остаются пустыми: `bot_safety.bot_safe_fields == []`.

Тест: `test_crm_card_uses_manager_only_call_summary_fallback_without_call_analysis`.

## Preview

Команда:

```bash
DB="/Users/dmitrijfabarisov/Projects/Mango_botsafe_summary_builder/.codex_local/botsafe_summary_builder_20260624/customer_timeline_wide_classfix_v4.sqlite"
OUT="/Users/dmitrijfabarisov/Projects/Mango_etap1_crm_cards/product_data/customer_timeline/crm_card_preview_20260624_d4_wide_classfix_v4_direct_fallback"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/build_crm_customer_card_workbook.py \
  --timeline-db "$DB" \
  --allowed-root "/Users/dmitrijfabarisov/Projects/Mango_botsafe_summary_builder/.codex_local/botsafe_summary_builder_20260624" \
  --out-xlsx "$OUT/crm_cards_preview.xlsx" \
  --tenant-id foton \
  --sample-size 200
```

Артефакты:

- `/Users/dmitrijfabarisov/Projects/Mango_etap1_crm_cards/product_data/customer_timeline/crm_card_preview_20260624_d4_wide_classfix_v4_direct_fallback/crm_cards_preview.xlsx`
- `/Users/dmitrijfabarisov/Projects/Mango_etap1_crm_cards/product_data/customer_timeline/crm_card_preview_20260624_d4_wide_classfix_v4_direct_fallback/crm_cards_preview.csv`
- `/Users/dmitrijfabarisov/Projects/Mango_etap1_crm_cards/product_data/customer_timeline/crm_card_preview_20260624_d4_wide_classfix_v4_direct_fallback/crm_cards_preview.summary.json`
- `/Users/dmitrijfabarisov/Projects/Mango_etap1_crm_cards/product_data/customer_timeline/crm_card_preview_20260624_d4_wide_classfix_v4_direct_fallback/direct_preview_quality_summary.json`
- `/Users/dmitrijfabarisov/Projects/Mango_etap1_crm_cards/product_data/customer_timeline/crm_card_preview_20260624_d4_wide_classfix_v4_direct_fallback/examples_10.json`
- `/Users/dmitrijfabarisov/Projects/Mango_etap1_crm_cards/product_data/customer_timeline/crm_card_preview_20260624_d4_wide_classfix_v4_direct_fallback/brand_mix_rows.json`
- `/Users/dmitrijfabarisov/Projects/Mango_etap1_crm_cards/product_data/customer_timeline/crm_card_preview_20260624_d4_wide_classfix_v4_direct_fallback/text_quality_blocking_examples.json`
- `/Users/dmitrijfabarisov/Projects/Mango_etap1_crm_cards/product_data/customer_timeline/crm_card_preview_20260624_d4_wide_classfix_v4_direct_fallback/amo_ai_field_ids_local_snapshot.json`

## Preview metrics

| Метрика | Значение |
|---|---:|
| Строк | 200 |
| `ready_yes` | 62 |
| `ready_no` | 138 |
| История общения непустая | 184 |
| Последняя сводка непустая | 184 |
| Raw JSON/service-id/мусор в видимых полях | 0 |
| Text-quality blocking rows | 19 |
| Text-quality warning rows | 86 |
| `ready_yes` с text-quality blocking | 2 |
| Бренд-смешение, все preview-строки | 2 |
| Бренд-смешение, только `ready_yes` | 0 |

Блокеры workbook:

| Блокер | Кол-во |
|---|---:|
| На телефоне несколько человек — проверьте, к кому относится | 91 |
| Не найдена сделка в AMO | 74 |
| Не найден контакт в AMO | 38 |

Text-quality blocking классы:

| Риск | Кол-во |
|---|---:|
| `empty_auto_history` | 16 |
| `completed_payment_next_step_conflict` | 1 |
| `wrong_person_or_identity_mismatch` | 1 |
| `closure_next_step_requires_downgrade` | 1 |

Комментарий по `ready_yes_text_quality_blocking=2`: обе строки не выглядят полезными для записи. Одна содержит сигнал `Контакт не подтвердился`/путаница с человеком, вторая — пассивный/закрывающий следующий шаг. Это полезные стопы gate, но текущий workbook ещё не переносит text-quality finding в `Готово=нет`.

## Brand check

Проверка считала только явные слова брендов `Фотон/Foton` и `УНПК/UNPK`; `МФТИ` не считался брендом, потому что в карточках это может быть площадка/контекст.

- `ready_yes`: 0 смешений.
- full preview: 2 смешения, оба `ready=нет`.

Остатки:

| Row | customer_id | brand | Причина | Готово | Блокер |
|---:|---|---|---|---|---|
| 29 | `customer:161c2420ed57c275f438a350193634de` | `unpk` | `foton_in_unpk` в старом событии истории | нет | ambiguous phone |
| 39 | `customer:1c8c3e4afdf4a03e07478357d183c1ca` | `foton` | `unpk_in_foton` в AMO title сделки | нет | ambiguous phone |

## 10 примеров preview

Сырой локальный файл с примерами: `/Users/dmitrijfabarisov/Projects/Mango_etap1_crm_cards/product_data/customer_timeline/crm_card_preview_20260624_d4_wide_classfix_v4_direct_fallback/examples_10.json`.

| Row | customer_id | brand | Готово | Блокер | Что видно |
|---:|---|---|---|---|---|
| 1 | `customer:0056897822baaaf456fc9a7b8219d9f8` | `unpk` | да | - | Последняя сводка и короткая история по звонку собраны |
| 2 | `customer:0152bccef9dcb4c1a4df15308f58864b` | `unknown` | да | - | Последняя сводка и хронология из нескольких касаний собраны |
| 3 | `customer:05f9d5e0bc65b858bafef2304cc91e28` | `unknown` | да | - | Есть последняя сводка и история с несколькими звонками |
| 4 | `customer:0604b62ae62127df016bda1291e6b59a` | `unknown` | да | - | Содержательный звонок попал в manager-only историю |
| 5 | `customer:069838b4de11441cb563fe705b97eb7e` | `unknown` | да | - | Сводка по формату/занятиям и история собраны |
| 6 | `customer:08bcb6c436d4ab9fb70e5f0b3bac4d00` | `unknown` | нет | ambiguous phone | Карточка строится, но запись должна быть заблокирована |
| 7 | `customer:099027984a147d927cbfd76dd2a831d7` | `unknown` | да | - | Сводка по B2B/школьному обращению собрана |
| 8 | `customer:0b43ed43be5e40a0cdfedfb7f981a3db` | `unpk` | да | - | Сводка по учебному вопросу и статусу занятия собрана |
| 9 | `customer:0be87264e87b17708c33b179caf6c855` | `unpk` | нет | ambiguous phone | Карточка строится, но есть text-quality риск оплаты/следующего шага |
| 10 | `customer:0c09eeb7bcf29119916efe532468c39e` | `unknown` | да | - | Сводка по актуальности летней школы собрана |

## AMO AI field_id, local read-only snapshot

Источник: локальные каталоги AMO, без обращения к live API. Перед live-write нужно подтвердить свежим AMO GET, потому что старый handoff export расходится со свежими catalog snapshots по двум contact AI-полям.

Contact, предпочтительно по свежим catalog snapshots:

| Field | field_id | type |
|---|---:|---|
| `Авто история общения` | 2359395 | textarea |
| `Статус матчинга` | 2360562 | text |
| `AI-приоритет` | 2360564 | text |
| `Последняя AI-сводка` | 2362743 | textarea |
| `AI-рекомендованный следующий шаг` | 2362745 | textarea |

Старый `prod_runtime_transfer` расходится:

| Field | old field_id | old type |
|---|---:|---|
| `Последняя AI-сводка` | 2360568 | text |
| `AI-рекомендованный следующий шаг` | 2360566 | text |

Lead/deal:

| Field | field_id | type |
|---|---:|---|
| `AI-рекомендованный следующий шаг` | 2362771 | textarea |
| `AI-дата следующего касания` | 2361382 | text |
| `AI-сводка по сделке` | 2361478 | textarea |
| `AI-история по сделке` | 2362773 | textarea |
| `AI-фактический статус сделки` | 2362775 | textarea |
| `AI-приоритет сделки` | 2362777 | text |
| `AI-актуальные возражения` | 2362779 | textarea |
| `AI-основание рекомендации` | 2362781 | textarea |
| `AI-качество привязки к сделке` | 2362783 | textarea |
| `AI-предупреждение по сделке` | 2362785 | textarea |
| `AI-Tallanto статус по сделке` | 2362787 | textarea |
| `AI-дата обновления сделки` | 2362789 | date_time |
| `AI-вердикт по закрытию` | 2361374 | text |
| `AI-risk: premature close` | 2361376 | text |
| `AI-основание вердикта` | 2361480 | textarea |

## Тесты

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_crm_card_aggregator.py
# 11 passed in 0.51s

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q
# 3381 passed, 5 skipped, 1 warning in 49.96s
```

## Остаточные риски

- Workbook `Готово` пока не учитывает text-quality findings: 2 строки `ready=да` должны быть остановлены gate на будущей записи.
- В полном preview есть 2 cross-brand попадания в `ready=нет` строках. Для будущего UX можно либо оставить как диагностический сигнал, либо отдельной правкой чистить/маскировать такие фрагменты в preview.
- Список `field_id` собран из локальных snapshots; перед live-write нужен свежий read-only GET каталога AMO.
- Вердикт на live/write не выносился.
