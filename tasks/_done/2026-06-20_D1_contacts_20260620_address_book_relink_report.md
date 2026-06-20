# D1: Fresh Tallanto Contacts Address Book Relink Preview

Дата: 2026-06-20  
Ветка: `codex/etap2-step1-address-book`  
Режим: read-only по Tallanto/timeline/CRM, без live-write.

## Что сделано

- `Contacts 20.06.2026.xls` сконвертирован через LibreOffice в UTF-8 CSV:
  `/Users/dmitrijfabarisov/Projects/mango-tz33-perf/_external_handoffs/tallanto_contacts_export_2026-06-20/converted/Contacts 20.06.2026.csv`
- Собрана новая адресная книга:
  `/Users/dmitrijfabarisov/Projects/mango-tz33-perf/_external_handoffs/tallanto_contacts_export_2026-06-20/identity_map/tallanto_email_identity_map.sqlite`
- Добавлена поддержка `ID Tallanto` как alias к `ID`.
- `Филиал`/Tallanto branch больше не используется как бренд для relink; бренд берётся только из email-события.
- Добавлен read-only Stage2 relink-preview для JSONL корпуса/дельты.

## Адресная книга

Свежий экспорт:

- строк: `11860`
- непустых ID: `11860`
- уникальных ID: `11802`
- строк с email: `11293`
- уникальных email-значений после нормализации: `11548`
- strong-unique email: `10797`
- строк с телефоном: `11760`
- уникальных телефонных значений: `11969`
- strong-unique телефон: `11157`
- повторяющихся Tallanto ID: `56`

Сравнение со старой картой 2026-05-12:

- старых клиентов: `18014`, новых клиентов: `11802`
- `+691` новых Tallanto ID
- `6903` старых Tallanto ID отсутствуют в свежем contacts-export
- новых контактных значений: `+1374`
- исчезнувших относительно старой карты значений: `14622`

Вывод: свежий contacts-export полезен для дозапривязки, но это не полная замена старой student-карты без отдельного решения владельца.

## Пересвязка Stage2

Корпус `27009`:

- вход: `13173` real correspondence + `13836` campaigns
- было привязано: `20444 / 27009` (`75.69%`)
- новая однозначная дозапривязка: `+651`
- стало привязано: `21095 / 27009` (`78.10%`)
- остаток unmatched: `5914`
- причины остатка:
  - `cross_brand_signal`: `2525`
  - `identity_value_missing`: `2383`
  - `duplicate_identity_value`: `980`
  - `multiple_identity_targets`: `15`
  - `duplicate_tallanto_id`: `11`

Дельта `3084`:

- вход: `1560` real + `1524` campaigns
- было привязано: `1134 / 3084` (`36.77%`)
- новая однозначная дозапривязка: `+630`
- стало привязано: `1764 / 3084` (`57.20%`)
- остаток unmatched: `1320`
- причины остатка:
  - `identity_value_missing`: `1188`
  - `no_identity_signal`: `75`
  - `cross_brand_signal`: `30`
  - `duplicate_identity_value`: `26`
  - `multiple_identity_targets`: `1`

Артефакты:

- audit pack: `/Users/dmitrijfabarisov/Projects/mango-tz33-perf/audits/_inbox/D1_contacts_20260620_relink_current_20260620201542`
- corpus report: `/Users/dmitrijfabarisov/Projects/mango-tz33-perf/_external_handoffs/mail_archive_2026-06-20/regru_edu/stage2_customer_relink_contacts_20260620/corpus_27009/mail_stage2_customer_relink_preview_report.json`
- corpus decisions: `/Users/dmitrijfabarisov/Projects/mango-tz33-perf/_external_handoffs/mail_archive_2026-06-20/regru_edu/stage2_customer_relink_contacts_20260620/corpus_27009/mail_stage2_customer_relink_preview_decisions.csv`
- delta report: `/Users/dmitrijfabarisov/Projects/mango-tz33-perf/_external_handoffs/mail_archive_2026-06-20/regru_edu/stage2_customer_relink_contacts_20260620/delta_3084/mail_stage2_customer_relink_preview_report.json`
- delta decisions: `/Users/dmitrijfabarisov/Projects/mango-tz33-perf/_external_handoffs/mail_archive_2026-06-20/regru_edu/stage2_customer_relink_contacts_20260620/delta_3084/mail_stage2_customer_relink_preview_decisions.csv`

## Примеры для регрейда

Сырые email/телефоны не пишутся в git-отчёт; ниже source line + хэши сигналов.

| set | source | line | decision | reason | old_match | brand | signal | message_sha | signal_hash |
|---|---:|---:|---|---|---|---|---|---|---|
| corpus | candidates.jsonl | 1 | linked | linked | ambiguous | unpk | email | 80237f4bb144c705 | 011df91f93b98d86 |
| corpus | candidates.jsonl | 20 | linked | linked | ambiguous | unpk | email | 2c2de829347d1345 | 531c0b389d1e7ce8 |
| corpus | candidates.jsonl | 53 | linked | linked | ambiguous | unknown | phone | ae061e89822b5784 | e358ae62a22f9963 |
| corpus | candidates.jsonl | 91 | linked | linked | ambiguous | unpk | email | 9534b94920eb54dc | 7ff87bede5cafc1f |
| corpus | candidates.jsonl | 235 | linked | linked | ambiguous | unknown | email | 1976546fafa237a0 | 1ca6998d20aa14a8 |
| corpus | candidates.jsonl | 2 | unmatched | duplicate_identity_value | ambiguous | unpk | - | 5e2e4b264140c3bb | - |
| corpus | candidates.jsonl | 3 | unmatched | identity_value_missing | ambiguous | unpk | - | 74c1d6e27421e5b9 | - |
| corpus | candidates.jsonl | 7 | unmatched | cross_brand_signal | none | unpk | - | bc625a6a8e4477ea | - |
| corpus | candidates.jsonl | 25 | unmatched | duplicate_identity_value | ambiguous | unknown | - | c1807f083fdb89f1 | - |
| corpus | candidates.jsonl | 30 | unmatched | identity_value_missing | none | unknown | - | dcff4a1fef772682 | - |
| delta | candidates.jsonl | 1 | linked | linked | none | unpk | email | 1f73cf0bf9f6175b | c378b8e3663f7d7c |
| delta | candidates.jsonl | 2 | linked | linked | none | unpk | email | c91098651b59360b | c378b8e3663f7d7c |
| delta | candidates.jsonl | 7 | linked | linked | none | unknown | email | 61acf2bc587c5bfc | 963e40317812e3e2 |
| delta | candidates.jsonl | 8 | linked | linked | none | unpk | email | 531000cf268b0a12 | ebf8730f84c0356b |
| delta | candidates.jsonl | 9 | linked | linked | none | unpk | email | d81c8b775317a2b9 | d9dcf93b9cbc52c6 |
| delta | candidates.jsonl | 4 | unmatched | duplicate_identity_value | none | unpk | - | ba9bd2d9dbca6e29 | - |
| delta | candidates.jsonl | 16 | unmatched | identity_value_missing | none | unpk | - | 19a77cb480bf17b1 | - |
| delta | candidates.jsonl | 42 | unmatched | identity_value_missing | none | foton | - | 85542c56cb6f59c8 | - |
| delta | candidates.jsonl | 45 | unmatched | identity_value_missing | none | unknown | - | b009d1630dd4d4ce | - |
| delta | candidates.jsonl | 74 | unmatched | identity_value_missing | none | unpk | - | e1989a115b5d3f18 | - |

## Проверки

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_productization_mail_archive.py` → `56 passed`
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests` → `3437 passed, 5 skipped`

## Semantic Review

Verdict: `PASS_WITH_NOTES`

Что прошло:

- Tallanto/timeline/CRM не изменялись.
- Бренд не берётся из `Филиал`; `brand_from_tallanto_trusted=false`.
- Общие email/телефоны и cross-brand сигналы блокируются.
- В git-отчёте нет сырых email/телефонов.

Остаточные риски:

- Свежий contacts-export неполный относительно старой student-карты: минус `6903` старых Tallanto ID.
- `unknown` brand события разрешают связь, если контактный сигнал уникален; это осознанный read-only preview, но для write-этапа нужен отдельный ручной gate.
- `cross_brand_signal` в корпусе большой (`2525`) — часть может быть реальной семейной/многобрендовой историей, но автоматически склеивать нельзя.

Рекомендация:

- Для timeline/write-этапа не заменять старую student-карту этой contacts-картой целиком. Использовать fresh contacts как дополнительный источник для дозапривязки или сначала собрать объединённую карту old+fresh с теми же предохранителями.
