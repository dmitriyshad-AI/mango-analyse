# AUDIT: AMO + Wappi integration, Phase 0

Дата: 2026-06-03

Автор проверки: Codex D3

Режим: строго read-only. В AMO/CRM/Tallanto/Wappi/TG/MAX ничего не создавал, не менял, не отправлял. Тестовое сообщение не слал. `stable_runtime` не изменял.

## Короткий вывод

Фаза 0 частично закрыта документально и локальными read-only снимками, но не закрыта live-проверкой текущего production AMO/Wappi.

Подтверждено:

- Wappi для Telegram и MAX присылает webhook `incoming_message` с `profile_id`, `chatId`, `from`, `to`, `body`, временем, типом сообщения и контактными полями.
- Wappi-страницы интеграции с amoCRM заявляют: переписка ведётся прямо в amoCRM, входящее сообщение нового клиента создаёт контакт и сделку, переписка хранится в CRM.
- amoCRM официально поддерживает webhook входящего сообщения с `chat_id`, `origin`, `text`, `element_id`; также есть read-only методы для контактов, сделок, событий, примечаний, воронок и связей чатов с контактами.
- В нашем AMO-снимке 2026-05-13 есть поля контакта для Telegram/MAX ID и username; это полезно для связи клиента, но не заменяет Wappi `profile_id`.
- Официального draft/prefill hook для вставки текста в поле существующего Wappi/amo-виджета не найдено. DOM-инъекцию закладывать нельзя.

Не подтверждено текущим доступом:

- Текущий live AMO-доступ на 2026-06-03: запрос к `https://api.fotonai.online/api/integrations/amocrm/status` с локально доступным ключом вернул `403 Forbidden`; ранее в этой же разведке наблюдался `502`. Значит сегодня live-read через AI Office не подтверждён.
- Реальные Wappi `profile_id` Фотон/УНПК в проектных env/config/docs не найдены.
- Реальная связка Wappi `chatId/profile_id` -> конкретная AMO сделка на наших данных не подтверждена без live-примера.
- Появление сообщения, отправленного через Wappi API, обратно в AMO-ленте не подтверждено read-only наблюдением; нужно отдельное разрешение на тест.

## Источники

Внешняя документация:

- Wappi Telegram API: `https://wappi.pro/telegram-api-documentation`
- Wappi MAX API: `https://wappi.pro/max-api-documentation`
- Wappi + amoCRM Telegram: `https://wappi.pro/integrations/amo/telegram`
- Wappi + amoCRM MAX: `https://wappi.pro/integrations/amo/max`
- amoCRM Webhooks format: `https://www.amocrm.ru/developers/content/crm_platform/webhooks-format`
- amoCRM Contacts API: `https://www.amocrm.ru/developers/content/crm_platform/contacts-api`
- amoCRM Leads API: `https://www.amocrm.ru/developers/content/crm_platform/leads-api`
- amoCRM Events and Notes: `https://www.amocrm.ru/developers/content/crm_platform/events-and-notes`
- amoCRM Pipelines API: `https://www.amocrm.ru/developers/content/crm_platform/leads_pipelines`
- amoCRM Web SDK Card: `https://www.amocrm.ru/developers/content/web_sdk/card`
- amoCRM Salesbot: `https://www.amocrm.ru/developers/content/digital_pipeline/salesbot`

Локальные read-only источники:

- `D1_audit_backlog/TZ_amo_wappi_integration_stage0_2026-06-03.md`
- `stable_runtime/deal_aware_amo_live_snapshot_20260513_v2/summary.json`
- `stable_runtime/deal_aware_amo_live_snapshot_20260513_v2/amo_status_catalog.csv`
- `stable_runtime/deal_aware_amo_live_snapshot_20260513_v2/amo_field_catalog.csv`
- `stable_runtime/deal_aware_amo_live_snapshot_20260513_v2/amo_contacts_snapshot.csv`
- `docs/AMO_AI_OFFICE_SERVER_ACCESS_2026-05-09.md`

## 1. Что приходит при входящем TG/MAX через Wappi

### Telegram

По официальной Wappi Telegram API документации входящее текстовое сообщение приходит в `messages[]` как объект с полями:

- `id`
- `profile_id`
- `wh_type = incoming_message`
- `timestamp`, `time`
- `body`
- `type`
- `from`, `to`
- `senderName`
- `chatId`
- `caption`
- `from_where`
- `contact_name`
- `contact_phone`
- `contact_username`
- `username`
- `is_forwarded`, `isReply`, `is_edited`, `is_me`
- `chat_type`
- `thumbnail`, `picture`
- `wappi_bot_id`
- `is_deleted`, `is_bot`

Для файлов/изображений/аудио дополнительно появляются `file_name`, `mimetype`, `file_link`, `file_link_expire` и похожие поля. В публичный отчёт не переношу примерные номера/имена из документации полностью, потому что для нашей архитектуры важна схема, а не тестовые значения.

Смысл для нас:

- `profile_id` - главный ключ канала и будущий главный сигнал бренда.
- `chatId` - идентификатор чата/пользователя в Telegram-канале Wappi.
- `from`/`to` - направление.
- `contact_phone` в Telegram может быть пустым, поэтому нельзя полагаться только на телефон.
- `contact_username`/`username` полезны как вспомогательный ключ, но не как единственный стабильный ключ.

### MAX

По официальной Wappi MAX API документации входящее текстовое сообщение приходит в `messages[]` как объект с полями:

- `id`
- `profile_id`
- `wh_type = incoming_message`
- `timestamp`, `time`
- `body`
- `type`
- `from`, `to`
- `senderName`, `senderLastName`
- `chatId`
- `caption`
- `from_where`
- `contact_name`, `contact_last_name`
- `contact_max_name`, `contact_max_last_name`
- `contact_phone`
- `contact_username`
- `phone`
- `is_forwarded`, `isReply`, `is_edited`, `is_me`
- `isGif`
- `thumbnail`, `picture`
- `wappi_bot_id`
- `is_deleted`, `is_bot`, `is_blacklist`

Смысл для нас:

- Для MAX `contact_phone` и `phone` в документации присутствуют, поэтому телефон может быть более сильным ключом, чем в Telegram.
- `chatId` в примере MAX не равен телефону; его нельзя автоматически считать номером клиента.
- `profile_id` остаётся ключом канала/бренда.

### Когда приходит

Wappi документирует несколько типов webhook-уведомлений:

- `incoming_message` - входящее сообщение.
- `outgoing_message_api` - исходящее через Wappi API.
- `outgoing_message_phone` - исходящее с телефона/аккаунта.
- `delivery_status` - статус доставки.
- `authorization_status` - статус профиля.
- `application_status` - статус приложения.

Для Этапа 0 нужно принимать только `wh_type = incoming_message`; все исходящие и системные события игнорировать или писать только в служебный read-only журнал.

### Приходит ли параллельно amo-событие и есть ли lead_id

Документально подтверждено:

- Wappi + amoCRM Telegram/MAX страницы заявляют, что сообщения можно вести из amoCRM, переписка хранится в CRM, а при входящем сообщении нового клиента создаются контакт и сделка.
- amoCRM webhook format содержит пример события входящего сообщения с `message.add`, где есть `chat_id`, `origin`, `text`, `element_id`, `element_type`.
- amoCRM Contacts API содержит read-only метод `GET /api/v4/contacts/chats`, который возвращает связь `chat_id` -> `contact_id`.

Не подтверждено на наших live-данных:

- какой именно `chat_id` видит amoCRM после Wappi-интеграции;
- совпадает ли он с Wappi `chatId`;
- всегда ли входящее Wappi-сообщение даёт amo `element_id` сделки;
- как выглядит реальный AMO event/message на нашей production-интеграции.

Вывод: в архитектуре нельзя заранее считать, что `Wappi chatId == amo chat_id` и что `lead_id` придёт напрямую. Это нужно проверить одним live-read/live-test циклом после отдельного разрешения Дмитрия.

## 2. Как стабильно связать Wappi chatId/profile_id с конкретной сделкой amo

Надёжная связь должна быть многошаговой и fail-closed.

### Разделить два разных идентификатора

`profile_id`:

- это профиль Wappi, то есть канал/аккаунт мессенджера;
- его нужно использовать как главный сигнал бренда;
- его нельзя использовать как идентификатор клиента.

`chatId`:

- это идентификатор чата/пользователя в конкретном канале;
- его нужно использовать для поиска клиента/контакта;
- его нельзя использовать как бренд.

### Рекомендуемый порядок поиска сделки

1. Определить бренд по `profile_id`.

   Если `profile_id` неизвестен, сразу `brand_fail_closed`: не генерировать клиентский черновик.

2. Попробовать связать чат с AMO-контактом через официальный AMO chat-link.

   Метод amoCRM: `GET /api/v4/contacts/chats` с `chat_id` или `contact_id`.

   Но это можно использовать только после проверки, что Wappi `chatId` совпадает или мапится на amo `chat_id`.

3. Если chat-link недоступен или не совпадает, искать контакт по AMO custom fields.

   В снимке AMO 2026-05-13 есть такие contact fields:

   - `Telegram ID` (`TGID`)
   - `Telegram username` (`TGUSERNAME`)
   - `Max ID` (`MAXID`)
   - `Max User ID` (`MAXUSERID`)
   - старые поля `TelegramId_WZ`, `TelegramUsername_WZ`, `MaxId_WZ`

   В `amo_contacts_snapshot.csv` непустые значения:

   - `Telegram ID`: 758 контактов, из них 750 со связанной сделкой.
   - `Telegram username`: 506 контактов, из них 501 со связанной сделкой.
   - `Max User ID`: 354 контакта, из них 348 со связанной сделкой.
   - `Max ID`: 2 контакта, из них 2 со связанной сделкой.
   - старые WZ-поля: `TelegramId_WZ` 17, `TelegramUsername_WZ` 15, `MaxId_WZ` 7.

   Это сильное доказательство, что AMO уже хранит channel identity, но не доказательство, что эти значения равны Wappi `chatId`.

4. Если есть телефон, искать контакт по нормализованному телефону.

   В снимке 2026-05-13: 12 229 контактов с телефоном из 12 700 контактов. Телефон - хороший вспомогательный ключ, но при дублях нужен запрет на авто-выбор.

5. Из найденного контакта получить связанные сделки.

   Для AMO Contacts API доступен `GET /api/v4/contacts/{id}?with=leads`; Leads API также возвращает `pipeline_id`, `status_id`, связанные контакты и источник.

6. Выбрать одну активную сделку.

   Если найдено 0 сделок, 2+ возможных сделок, закрытая сделка без явного правила, конфликт бренда или конфликт канала - статус `manual_resolution_required`, без клиентского текста.

### Критичное правило

Бренд нельзя выводить из телефона, имени, текста вопроса или названия сделки. Бренд сначала идёт из Wappi `profile_id`, а AMO `pipeline_id` может только подтвердить или заблокировать.

## 3. Появляется ли сообщение, отправленное через Wappi API, в ленте amo

Документально подтверждено:

- Wappi Telegram/MAX API документирует webhook `outgoing_message_api` с `profile_id`, `body`, `chatId`, `from_where = api`, `task_id`.
- Wappi + amoCRM Telegram/MAX страницы заявляют, что переписка хранится в CRM и что менеджер может писать из карточки/чата amoCRM через подключённый Wappi-профиль.

Не подтверждено на наших live-данных:

- если наш сервис отправит сообщение через Wappi API напрямую, появится ли оно в AMO-ленте этой же сделки;
- будет ли это именно сообщение чата, примечание, событие или другой объект;
- с какой задержкой появится readback;
- какой ID будет общим между Wappi webhook и AMO timeline.

Вывод для Этапа 0:

- Не использовать Wappi send API.
- Не обещать метрику "менеджер отправил без правки" через AMO-ленту, пока нет live-readback теста.
- Для будущего human-click send нужен отдельный тест: отправка в тестовый контакт через Wappi API -> read-only проверка AMO ленты -> сверка текста и ID. Этот тест является live-send и требует отдельного OK Дмитрия.

## 4. Есть ли официальный draft/prefill hook

Подтверждено отрицательно: официального метода "вставить черновик в поле существующего Wappi/amo compose" в найденных документах не обнаружено.

Что есть официально:

- amoCRM Web SDK позволяет делать свой виджет в карточке, правой колонке, списках и интерфейсе карточки.
- amoCRM Card Interface поддерживает click-to-call / "написать первым" через зарегистрированные каналы.
- amoCRM Salesbot умеет отправлять сообщения клиенту.
- amoCRM Notes API умеет читать и добавлять примечания.
- Wappi-виджет позволяет писать из amoCRM, но найденные публичные документы не дают API для prefill чужого поля ввода.

Что это значит:

- DOM-инъекцию в Wappi/amo UI не закладывать.
- На Этапе 0 безопасный вариант - локальный dry-run журнал.
- После отдельного OK можно добавить внутреннюю AMO note с черновиком, но это уже write-операция.
- На следующем этапе лучший официальный интерфейс - собственный amoCRM widget: показать черновик, кнопка "скопировать", "отклонить", "обновить". Это не prefill Wappi-поля, но устойчивый и официальный путь.
- Human-click send через Wappi API - отдельная фаза с kill-switch, журналом и readback.

## 5. Бренд-карты

### Wappi profile_id -> бренд

Статус: не заполнено, нет достаточных данных.

В проекте не найдены реальные значения Wappi `profile_id` для Фотон/УНПК. В ТЗ карта пока placeholder:

| Канал | profile_id | Бренд | Статус |
|---|---:|---|---|
| Telegram Фотон | неизвестно | foton | нужно взять из Wappi кабинета или live webhook |
| MAX Фотон | неизвестно | foton | нужно взять из Wappi кабинета или live webhook |
| Telegram УНПК | неизвестно | unpk | нужно взять из Wappi кабинета или live webhook |
| MAX УНПК | неизвестно | unpk | нужно взять из Wappi кабинета или live webhook |

Как получить безопасно:

- read-only список профилей Wappi из кабинета/API, если Дмитрий даст Wappi token;
- либо один заранее согласованный тестовый входящий webhook по каждому профилю;
- либо скрин/экспорт настроек Wappi с названием профиля, `profile_id`, типом канала и брендом.

### amo pipeline_id -> бренд

Статус: AMO pipeline IDs известны, бренд не подтверждён.

Из `amo_status_catalog.csv` snapshot 2026-05-13:

| pipeline_id | pipeline_name по статус-каталогу | Статус бренд-карты |
|---:|---|---|
| 8938034 | Лиды | бренд не подтверждён |
| 10408062 | Сделки B2C | бренд не подтверждён |
| 10431046 | Обзвон | бренд не подтверждён |

В `amo_deals_snapshot.csv` есть неоднозначность имён pipeline:

- `8938034` встречается как `Обзвон` и `Лиды`;
- `10408062` встречается как `Сделки B2C` и `Обзвон`;
- `10431046` встречается как `Обзвон`.

Вывод: для кода использовать только `pipeline_id` как стабильный ключ. `pipeline_name` из deal snapshot нельзя считать источником правды. Даже `pipeline_id` нельзя автоматически считать брендом, пока Дмитрий/РОП не подтвердят карту Фотон/УНПК.

Рекомендованная временная карта:

| pipeline_id | brand | rule |
|---:|---|---|
| 8938034 | unknown | только подтверждающий сигнал запрещён |
| 10408062 | unknown | только подтверждающий сигнал запрещён |
| 10431046 | unknown | только подтверждающий сигнал запрещён |

До подтверждения карты AMO pipeline не должен ни выбирать бренд, ни разблокировать брендовый конфликт.

## Что подтверждено против пяти вопросов Фазы 0

| Вопрос | Ответ | Уровень подтверждения |
|---|---|---|
| Что приходит при входящем TG/MAX через Wappi | Схемы payload подтверждены документацией Wappi | docs_confirmed |
| Приходит ли amo-событие параллельно | В документации amoCRM есть `message.add`; Wappi заявляет хранение переписки и создание контакта/сделки | docs_confirmed, real_sample_missing |
| Есть ли связь с lead_id | В amo webhook есть `element_id`, Wappi создаёт сделку для нового клиента, но реальная связка не проверена | not_confirmed_live |
| Как связать chat/profile с AMO deal | Через `profile_id` для бренда, chat-link/custom fields/phone для контакта, затем linked leads | design_confirmed, mapping_needs_live |
| Wappi API outbound появляется в AMO | Документация Wappi показывает outgoing webhook, Wappi+AMO заявляет хранение переписки; live readback не проверен | likely_but_needs_live_test |
| Официальный draft/prefill hook | Не найден; закладывать нельзя | negative_confirmed_by_docs_search |
| Бренд-карты | Wappi profile_id неизвестны; AMO pipeline IDs известны, бренды не подтверждены | incomplete |

## LIVE-подтверждение 2026-06-04

Режим проверки: строго read-only. Использовались только `GET`-маршруты AI Office API и локальные snapshot-файлы как указатели для выбора кандидатов. AMO/CRM/Tallanto/Wappi/TG/MAX write/send не выполнялись. Тестовое сообщение не отправлялось. ПДн в этом разделе замаскированы.

### Live API статус

Подтверждено:

- `GET https://api.fotonai.online/api/health` -> `200`, `status=ok`, `service=api`.
- `GET https://api.fotonai.online/api/openapi.json` -> `200`.
- `GET https://api.fotonai.online/api/integrations/amocrm/status` с `X-API-Key` -> `200`.
- amoCRM live status: `connected=true`, `status=active`, `account_base_url=https://educent.amocrm.ru`, `token_source=oauth`, `contact_field_count=27`, `required_contact_fields_missing=[]`.

Доступные read-only AMO маршруты в текущем AI Office API:

- `GET /api/integrations/amocrm/status`
- `GET /api/integrations/amocrm/contacts/by-phone`
- `GET /api/integrations/amocrm/leads/by-phone`

Отдельных live read-only маршрутов для AMO `chats`, `events`, `notes`, `pipelines` в текущем опубликованном API нет. Поэтому `GET /api/v4/contacts/chats` напрямую не проверен: у меня нет прямого AMO access token, а AI Office сейчас не проксирует этот метод.

### Реальная AMO бренд-карта

Live + snapshot evidence показывает: AMO `pipeline_id` сейчас не является брендом. Воронки отражают процесс.

Доказательство:

- Snapshot 2026-05-13 содержит процессные воронки: `8938034 Лиды`, `10408062 Сделки B2C`, `10431046 Обзвон`.
- Live-проверка двух разных брендовых кейсов показала, что оба находятся в одной и той же воронке и статусе:
  - Telegram-кейс с бренд-маркерами `УНПК/МФТИ`: `pipeline_id=10408062`, `status_id=83489762`.
  - MAX-кейс с бренд-маркером `Фотон`: `pipeline_id=10408062`, `status_id=83489762`.

Вывод: `pipeline_id=10408062` значит `Сделки B2C`, а не Фотон или УНПК. Бренд нельзя определять по pipeline.

Где реально живёт бренд:

| Поле | Статус | Live/snapshot evidence |
|---|---|---|
| Lead custom field `Организация` | главный брендовый признак | live TG-кейс дал `unpk_or_mfti`; live MAX-кейс дал `foton` |
| Lead custom field `Филиал` | дополнительный признак | live TG-кейс дал `unpk_or_mfti` |
| `utm_campaign` | дополнительный признак из snapshot | snapshot: Фотон-маркеры 145, УНПК/МФТИ 2 |
| `utm_term` | дополнительный признак из snapshot | snapshot: УНПК/МФТИ 103 |
| Contact field `Филиал Tallanto` | дополнительный, не основной признак | snapshot: 9127 заполнено, МФТИ-маркер 1505 |
| Landing `/FOTON` / `/UNPK` | логически важный признак, но в AMO snapshot отдельного поля landing не найдено | нужен Wappi/site-source readback |
| Tags | в snapshot нет отдельной tag-колонки | нужен live endpoint или AMO export |

Рекомендация для resolver:

1. Главный бренд входящего сообщения всё равно должен идти из Wappi `profile_id`.
2. AMO `Организация` и `Филиал` использовать как подтверждающий сигнал.
3. AMO pipeline использовать только как процессный контекст, не как бренд.
4. Если `profile_id` и AMO brand fields конфликтуют, делать `brand_fail_closed`.

### Wappi profile_id -> бренд

Live Wappi profile list не получен: в проекте не найден Wappi API token или сохранённые `profile_id`.

Подтверждено по документации Wappi:

- Telegram profile list: `GET /tapi/profile/all/get` с заголовком `Authorization`.
- MAX profile list: `GET /maxapi/profile/all/get` с заголовком `Authorization`.

Нужно от Дмитрия:

- Wappi API token для read-only вызова списка профилей, либо ручная таблица/скрин:
  - `profile_id -> канал -> бренд`;
  - Фотон Telegram;
  - Фотон MAX;
  - УНПК Telegram;
  - УНПК MAX.

До этого `WAPPI_PROFILE_BRAND` остаётся незаполненным и production resolver обязан fail-closed при неизвестном `profile_id`.

### Live identity chain: Telegram

Live-проверка по одному реальному Telegram-контакту:

| Шаг | Результат |
|---|---|
| Входной ключ | телефон использовался только внутри запроса; в отчёте: `+7***5595` |
| Snapshot contact | `contact_id=76093890`, ровно 1 linked lead |
| Snapshot channel fields | `Telegram ID`, `Telegram username` |
| Live `contacts/by-phone` | `200`, `status=ok`, `count=1` |
| Live contact id | `76093890`, совпадает со snapshot |
| Live channel fields | `Telegram ID`, `Telegram username`, `Телефон` |
| Live linked leads in contact | `1` |
| Live `leads/by-phone` | `200`, `status=matched`, `contact_count=1`, `lead_count=1` |
| Live lead | `lead_id=48593774`, `pipeline_id=10408062`, `status_id=83489762`, embedded contact count `1` |
| Brand fields | `Организация=unpk_or_mfti`, `Филиал=unpk_or_mfti` |

Подтверждённая цепочка: телефон -> AMO contact -> Telegram identity fields -> linked AMO lead -> brand fields.

Не подтверждено: Wappi `chatId` -> AMO contact через `GET /api/v4/contacts/chats`, потому что текущий AI Office API не даёт этот маршрут.

### Live identity chain: MAX

Live-проверка по одному реальному MAX-контакту:

| Шаг | Результат |
|---|---|
| Входной ключ | телефон использовался только внутри запроса; в отчёте: `+7***2487` |
| Snapshot contact | `contact_id=66354037`, ровно 1 linked lead |
| Snapshot channel fields | `Max User ID` |
| Live `contacts/by-phone` | `200`, `status=ok`, `count=1` |
| Live contact id | `66354037`, совпадает со snapshot |
| Live channel fields | `Max User ID`, `Телефон` |
| Live linked leads in contact | `1` |
| Live `leads/by-phone` | `200`, `status=matched`, `contact_count=1`, `lead_count=1` |
| Live lead | `lead_id=49284055`, `pipeline_id=10408062`, `status_id=83489762`, embedded contact count `1` |
| Brand fields | `Организация=foton` |

Подтверждённая цепочка: телефон -> AMO contact -> MAX identity field -> linked AMO lead -> brand field.

Не подтверждено: Wappi `chatId` -> AMO contact через `GET /api/v4/contacts/chats`, потому что текущий AI Office API не даёт этот маршрут.

### Как реально выглядит входящее сообщение + amo-событие

Остаётся не подтверждено live-наблюдением.

Причина:

- Wappi token/profile list отсутствует.
- Тестовое сообщение не отправлялось по условиям задачи.
- Текущий AI Office API не публикует AMO read-only routes для `events`, `notes`, `chats`.
- Прямого AMO access token в локальной среде нет; production OAuth хранится на стороне AI Office.

Что подтверждено документально:

- Wappi webhook `incoming_message` содержит `profile_id`, `chatId`, `body`, `from`, контактные поля.
- amoCRM поддерживает chat/message events и связь чата с контактом через `contacts/chats`.

Что нужно для полного закрытия:

1. Добавить в AI Office строго read-only endpoint для:
   - `GET /api/v4/contacts/chats`;
   - `GET /api/v4/events` с фильтром по contact/lead/chat;
   - `GET /api/v4/leads/{lead_id}/notes` только чтение.
2. Или дать прямой временный AMO read-only token/роль для проверки этих трёх методов.
3. Или дать Wappi webhook delivery log по существующему сообщению без отправки нового теста.

### Обновлённая brand_and_identity_resolution_matrix

| Сигнал | Что подтверждено live | Использование | Статус |
|---|---|---|---|
| Wappi `profile_id` | не получен, токена Wappi нет | главный бренд канала | blocked_by_missing_wappi_token |
| Wappi `chatId` | не получен на реальном payload | ключ чата, должен мапиться на contact/chat | blocked_by_missing_wappi_payload |
| AMO `Telegram ID` / `Telegram username` | live TG contact содержит поля | стабильный channel identity для AMO contact | confirmed |
| AMO `Max User ID` | live MAX contact содержит поле | стабильный channel identity для AMO contact | confirmed |
| Телефон | live lookup нашёл TG и MAX contacts | fallback key для поиска contact | confirmed, но не основной |
| AMO contact -> linked lead | TG/MAX live contacts имеют ровно 1 linked lead | связь contact -> deal | confirmed |
| AMO lead `Организация` | live TG=`unpk_or_mfti`, live MAX=`foton` | подтверждение бренда | confirmed |
| AMO lead `Филиал` | live TG=`unpk_or_mfti` | дополнительное подтверждение бренда | confirmed_partial |
| AMO `pipeline_id` | обе live-сделки разных брендов в `10408062` | процесс, не бренд | confirmed_not_brand |
| AMO `contacts/chats` | не проверен | желательный прямой chatId bridge | blocked_by_missing_endpoint |
| AMO events/notes | не проверены | readback истории сообщений | blocked_by_missing_endpoint |

### Остаток после live-проверки

Критичные блокеры перед реализацией Wappi Phase 1:

1. Нет Wappi API token или ручной таблицы `profile_id -> канал -> бренд`.
2. Нет live-примера Wappi `incoming_message` payload из существующей истории.
3. Текущий AI Office API не даёт read-only доступ к `contacts/chats`, `events`, `notes`.
4. Не проверено, появляется ли Wappi outbound API message в AMO ленте. Это live-send тест, его нельзя делать без отдельного OK Дмитрия.

Разумный следующий read-only шаг:

- добавить или временно включить в AI Office три read-only маршрута AMO: `contacts/chats`, `events`, `lead notes`;
- получить Wappi token только для `profile/all/get`;
- после этого обновить этот отчёт финальным `LIVE_VERDICT=phase0_closed`.

## Остаток вопросов для Дмитрия

1. Дать Wappi `profile_id` для четырёх профилей: Фотон Telegram, Фотон MAX, УНПК Telegram, УНПК MAX.
2. Подтвердить, есть ли отдельные AMO pipeline для Фотон/УНПК или pipeline сейчас отражают процесс, а не бренд.
3. Дать тестовую сделку/контакт, где можно read-only смотреть историю Wappi-сообщений.
4. Разрешить или запретить один тестовый live-send через Wappi в тестовый контакт для проверки readback в AMO.
5. Дать read-only доступ к Wappi profile list/API или скрин/экспорт профилей.
6. Подтвердить, можно ли на следующем этапе писать внутреннюю AMO note в тестовую карточку. Без этого Этап 0 остаётся только локальным dry-run журналом.

## Рекомендация по следующему шагу

Следующий безопасный шаг без live-write:

1. Получить Wappi profile list read-only или ручную таблицу `profile_id -> канал -> бренд`.
2. Получить один существующий AMO contact/lead с Telegram и один с MAX, где уже есть история сообщений.
3. Только читать:
   - AMO contact fields;
   - linked leads;
   - notes/events;
   - если доступно, `GET /api/v4/contacts/chats`.
4. Составить `brand_and_identity_resolution_matrix`: какие ключи реально совпадают (`Wappi chatId`, AMO chat_id, Telegram ID, Max User ID, phone).
5. Только после этого переходить к коду Этапа 0.

## Safety checklist

- AMO write: не выполнялся.
- Wappi send: не выполнялся.
- Telegram/MAX send: не выполнялся.
- Tallanto: не трогал.
- ASR/R+A: не запускал.
- stable_runtime DB/audio/transcripts: не изменял.
- Персональные данные в отчёте: телефоны, имена клиентов и реальные channel IDs не вынесены.
