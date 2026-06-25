# Wappi -> AMO Authoritative Auto Resolver: Step 0 STOP

Дата: 2026-06-25
Ветка: `codex/wappi-authoritative-auto-resolver`
База: `codex/wappi-botsafe-memory @ 201b30a`
ТЗ: `/Users/dmitrijfabarisov/Claude Projects/Foton/2026-06-25_TZ_Wappi_AMO_authoritative_auto_resolver.md`

## Вердикт

STOP. В raw Wappi payload на проверенной выборке нет пригодной авторитетной
связки `chat -> AMO lead/contact`.

Код resolver не менялся. Старый resolver не сломан и остаётся fallback.

## Что Проверено Read-Only

Проверка выполнена только через raw API payload:

- Wappi GET profiles/chats/messages по 4 профилям:
  - Foton Telegram: `ec2eed50-b55f`
  - Foton Max: `2952990f-9e4c`
  - UNPK Telegram: `18b255b8-7a67`
  - UNPK Max: `152b441d-81a2`
- AMO read-only GET через MCP только для известных ручных пар:
  `/leads/{lead_id}?with=contacts`.

Не использовался UI. Секреты/токены не печатались.

## Итог По Authoritative Link

| Проверка | Результат |
|---|---:|
| Wappi-чаты в основной выборке | 20 |
| Профили / каналы | 4 / `foton telegram`, `foton max`, `unpk telegram`, `unpk max` |
| Dialog payload с прямым `lead_id/contact_id/entity_id/amo_lead_id/amo_contact_id` | 0 |
| Dialog `item_link` / `item_name` непустые | 0 |
| Message `crm_entities` присутствует | да, в сообщениях |
| Message `crm_entities` непустой | 0 |
| Authoritative accepted | 0 |
| Authoritative rejected | 20 |
| Осталось только на старом fallback | 20 |

Сырые признаки:

- Max dialog имеет поля `item_link`, `item_name`, но в выборке они пустые.
- Telegram dialog не содержит прямого AMO lead/contact id.
- Message payload содержит `crm_entities`, но значения пустые:
  `{"chat_id": "", "crm_id": "", "crm_type": "", "manager_id": "", "message_id": ""}`.
- В известных ручных парах deeper scan на 47 сообщениях также дал
  `crm_entities_nonempty=0`.

## Known Manual Pairs: Дополнительная Проверка

| Профиль | Chat | Lead | Wappi dialog link | Messages checked | Non-empty `crm_entities` | AMO GET lead with contacts |
|---|---|---|---|---:|---:|---|
| UNPK Telegram `18b255b8-7a67` | `674***79` | `493***89` | empty | 19 | 0 | active, 1 linked contact |
| Foton Telegram `ec2eed50-b55f` | `290***69` | `478***47` | empty | 13 | 0 | closed, 1 linked contact |
| Foton Telegram `ec2eed50-b55f` | `605***82` | `497***41` | empty | 15 | 0 | active, 1 linked contact |

Вывод: AMO знает связь lead-contact, но Wappi raw payload не отдаёт связь
chat-message -> AMO lead/contact.

## 20 Masked Rejected Examples

Причина везде: `no_authoritative_link_in_raw_payload`.

| # | Brand | Channel | Profile | Chat | Raw evidence |
|---:|---|---|---|---|---|
| 1 | foton | max | `2952990f-9e4c` | `161***45` | dialog `item_link=""`; messages absent in sample |
| 2 | foton | max | `2952990f-9e4c` | `411***75` | dialog `item_link=""`; message `crm_entities` empty |
| 3 | foton | max | `2952990f-9e4c` | `173***65` | dialog `item_link=""`; message `crm_entities` empty |
| 4 | foton | max | `2952990f-9e4c` | `232***30` | dialog `item_link=""`; message `crm_entities` empty |
| 5 | foton | max | `2952990f-9e4c` | `651***17` | dialog `item_link=""`; message `crm_entities` empty |
| 6 | foton | telegram | `ec2eed50-b55f` | `195***79` | no dialog AMO id; message `crm_entities` empty |
| 7 | foton | telegram | `ec2eed50-b55f` | `290***69` | no dialog AMO id; message `crm_entities` empty |
| 8 | foton | telegram | `ec2eed50-b55f` | `822***54` | no dialog AMO id; message `crm_entities` empty |
| 9 | foton | telegram | `ec2eed50-b55f` | `154***33` | no dialog AMO id; message `crm_entities` empty |
| 10 | foton | telegram | `ec2eed50-b55f` | `204***66` | no dialog AMO id; message `crm_entities` empty |
| 11 | unpk | max | `152b441d-81a2` | `129***96` | dialog `item_link=""`; message `crm_entities` empty |
| 12 | unpk | max | `152b441d-81a2` | `136***87` | dialog `item_link=""`; message `crm_entities` empty |
| 13 | unpk | max | `152b441d-81a2` | `205***61` | dialog `item_link=""`; message `crm_entities` empty |
| 14 | unpk | max | `152b441d-81a2` | `200***67` | dialog `item_link=""`; message `crm_entities` empty |
| 15 | unpk | max | `152b441d-81a2` | `262***73` | dialog `item_link=""`; message `crm_entities` empty |
| 16 | unpk | telegram | `18b255b8-7a67` | `733***88` | no dialog AMO id; message `crm_entities` empty |
| 17 | unpk | telegram | `18b255b8-7a67` | `364***61` | no dialog AMO id; message `crm_entities` empty |
| 18 | unpk | telegram | `18b255b8-7a67` | `378***12` | no dialog AMO id; message `crm_entities` empty |
| 19 | unpk | telegram | `18b255b8-7a67` | `116***60` | no dialog AMO id; message `crm_entities` empty |
| 20 | unpk | telegram | `18b255b8-7a67` | `693***27` | no dialog AMO id; message `crm_entities` empty |

Accepted examples: 0, потому что API не отдал прямую связь.

## Сравнение Со Старым Resolver

Старый resolver был прогнан read-only на той же форме выборки: Wappi GET +
AMO GET, без создания pairs и без note write.

| Outcome старого fallback | Count |
|---|---:|
| `matched:Telegram ID` | 3 |
| `rejected:multi_active_lead` | 3 |
| `rejected:no_active_lead` | 1 |
| `rejected:max_phone_missing` | 6 |
| `rejected:closed_lead` | 6 |
| `rejected:brand_mismatch` | 1 |

P0 mismatch `старый resolver выбрал X, authoritative link показывает Y`: 0,
потому что authoritative `Y` отсутствует в raw payload.

## Что Не Делалось

- Код resolver не менялся.
- Auto-pairs не создавались.
- Основные `~/.mango_secrets/amo_wappi_profiles.json` и
  `~/.mango_secrets/draft_loop_pairs.json` не менялись.
- Wappi `mark_all=false`; клиентам ничего не отправлялось.
- AMO writes = 0.
- Tallanto/CRM writes = 0.
- Live Telegram bot / stable_runtime не трогались.
- Live-write заметок не запускался.

## Следующий Возможный Шаг

Без нового API-поля от Wappi/AI Office прямой authoritative resolver строить
нельзя. Нужно либо:

1. получить от Wappi/AI Office endpoint/поле, где `chat_id` явно связан с
   `amo_lead_id` или `amo_contact_id`;
2. либо оставить текущий осторожный fallback resolver как единственный
   автоматический способ, с запретом на имя/частичный телефон/мульти-сделки.

До появления такого сырья live-write через auto-resolver расширять нельзя.
