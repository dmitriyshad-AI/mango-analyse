# ТЗ-05. Один версионированный публикатор рабочей Google-таблицы

Статус: проект, реализация и изменения Google запрещены до отдельного подтверждения Дмитрия.

Зависимости: финальные контракты ТЗ-01…ТЗ-04.

## 1. Простыми словами

Сейчас Google заполняет длинная текстовая автоматизация. Она умеет многое, но её бизнес-логика не является обычным проверяемым кодом. Кроме того, `sync_status=done` означает только «когда-то публиковали», а не «строка соответствует нынешнему анализу».

Нужен один небольшой публикатор в репозитории. Каждый запуск он заново строит желаемые 16 колонок, сравнивает их со всем листом, исправляет только отсутствующие/устаревшие строки, сортирует всю строку целиком и перечитывает результат. Автоматизация лишь запускает этот скрипт.

Для связи звонка со строкой нужны два маленьких технических реестра в той же SQLite. Это не новая база и не второй конвейер: call-ledger хранит стабильный call key/номер/хэш, а incidents помнит неопознанную физическую строку и её SLA без бизнес-текста. Полная сверка Google остаётся обязательной — реестры не заменяют readback.

## 2. Доказательства проблемы

Read-only аудит стабильного снимка:

- Google `678` строк = `678` `sync_done` в SQLite;
- `673` строки однозначно связались по бизнес-идентичности, `669` — по точному хэшу полной расшифровки;
- `620/673` конспектов совпали с текущим собственным Analyse;
- `53` были устаревшей/другой исторической проекцией своего звонка;
- `0` конспектов точно совпали с Analyse другого звонка — массовый row shift не доказан;
- видимые A:P не содержат стабильный `source_call_id`, а временная Q очищается после сортировки;
- `CallRecord` имеет только `sync_status`, без версии анализа/проекции/readback;
- исторический живой publisher находился в natural-language automation `<owner-home>/.codex/automations/mango-calls-google-sheets/automation.toml`, а checked-in `scripts/publish_current_mango_calls_google.py` работал с другой 19-колоночной предварительной схемой и без полной расшифровки.

Следовательно, проблема конспектов вызвана прежде всего stale/разноверсионной проекцией и недоказанным Analyse, а не доказанным сдвигом сортировки. Но без стабильного ledger нельзя строго исключить сдвиг для каждой будущей записи.

## 3. Что уже работает и должно сохраниться

Из текущей автоматизации переиспользуются как обязательные acceptance requirements:

- ровно один лист `Звонки`, 16 видимых колонок A:P;
- UTC → `Europe/Moscow` ровно один раз;
- длительность `N мин M с`: для конечного `duration_sec >= 0` единая формула `whole_seconds = floor(duration_sec + 0.5)` («0,5 вверх»), затем `minutes, seconds = divmod(whole_seconds, 60)`; встроенный языковой `round()` с банковским округлением запрещён;
- полная расшифровка обязательна;
- самые свежие звонки сверху по точному `(started_at_utc, stable tie-breaker)`;
- append через native append/appendCells, не `count+row`;
- полный multiset/readback, Q очищена;
- append/update, заполнение Q, `sortRange`, очистка Q и layout выполняются одним атомарным Google `spreadsheets.batchUpdate`; отдельной мутации между Q и clear нет;
- J `WRAP/TOP`, P `CLIP/TOP`;
- ширина J фиксируется версионированной константой `SUMMARY_COLUMN_WIDTH_PX`; перед первым включением берётся текущая ширина листа, проходит визуальное подтверждение Дмитрия и после этого восстанавливается каждым запуском;
- высота каждой строки только по J, не по полной расшифровке P:
  - `logical_lines = sum(max(1, ceil(len(paragraph)/85)))`;
  - `pixelSize = min(260, max(42, 12 + 18*logical_lines))`;
  - `autoResizeDimensions` для строк запрещён;
- формульная защита всех текстовых полей.

Эти требования уже присутствуют в automation prompt и не должны реализовываться вторым несовместимым способом.

Официальная опора API: Google гарантирует, что все корректные подзапросы одного [`spreadsheets.batchUpdate`](https://developers.google.com/workspace/sheets/api/reference/rest/v4/spreadsheets/batchUpdate) применяются вместе атомарно, но прямо предупреждает о возможных последующих правках соавторов — поэтому exact readback всё равно обязателен. [Лимиты Sheets API](https://developers.google.com/workspace/sheets/api/limits) задают минутные квоты, рекомендуют payload до 2 МБ и ограничивают обработку одного запроса 180 секундами; эти значения читаются как внешние границы, а не как повод ослаблять идемпотентность.

## 4. Единственный писатель

Добавляется один checked-in скрипт, например `scripts/publish_live_mango_calls_google.py`.

- Он является единственным владельцем A:P рабочего листа.
- Automation только запускает скрипт, читает структурированный итог и не формирует строки сама.
- Старый natural-language writer сначала переводится в read-only shadow, затем отключается только после доказанного нового GO.
- Старый 19-колоночный pilot script не расширяется до production и не пишет в этот лист.
- Capture, Whisper, GigaAM, Resolve и Analyse не останавливаются; publisher читает WAL и пишет только собственный ledger/sync state.
- Один локальный `flock` по destination запрещает наложение двух запусков на текущем Mac. Многомашинный distributed writer не разрешён этим ТЗ.

## 5. Стабильный call key и ledger

`call_key = stable_event_key(tenant_id, provider, provider_call_id)` по существующему productization-контракту. Для текущего сервиса tenant=`mango`, provider=`mango_office`, provider_call_id=`source_call_id`. Локальный `id`, телефон, filename и номер строки не являются ключом.

`destination_id` строится единственным способом: `google_sheets:v1:<spreadsheet_id>:<numeric_sheet_id>:<tenant_id>:<environment>`. Для рабочего листа `numeric_sheet_id` — устойчивый `sheetId`, а не изменяемое имя вкладки; `environment` — закрытый enum `production|staging|test`. Пустой/неизвестный компонент блокирует запуск. Формула не меняется без миграции ledger и version bump.

Перед включением обязательны preflight: non-empty/unique call key, отсутствие конфликтов provider IDs и точное сопоставление каждой существующей физической строки; unresolved `identity_quarantine` блокирует включение.

В той же working SQLite добавляется основная call-ledger таблица:

```text
google_publication_ledger
  destination_id              TEXT
  call_key                     TEXT
  display_number               INTEGER NOT NULL CHECK(display_number > 0)
  projection_version           TEXT
  source_fingerprint           TEXT
  planned_row_sha256           TEXT|null
  last_verified_row_sha256     TEXT|null
  status                       reserved|verified|error|identity_quarantine|data_quarantine
  attempts                     INTEGER
  last_error_code              TEXT
  next_retry_at                UTC timestamp|null
  reserved_at                  UTC timestamp|null
  published_at                 UTC timestamp|null
  verified_at                  UTC timestamp|null
  resolution_by               TEXT|null
  resolution_at               UTC timestamp|null
  resolution_reason_code      TEXT|null
  PRIMARY KEY(destination_id, call_key)
  UNIQUE(destination_id, display_number)
```

Таблица не хранит телефон, transcript, summary или физический номер строки. `display_number` — неизменяемое бизнес-значение видимой колонки A, а не позиция строки и не identity звонка. При bootstrap сохраняется существующий уникальный положительный A; новому звонку номер назначается как следующий свободный положительный номер в детерминированном порядке batch. После полного identity gate, но до Google-write, короткая `BEGIN IMMEDIATE` фиксирует `call_key + display_number + source_fingerprint + planned_row_sha256` со статусом `reserved`; `sync_status` остаётся pending. Сортировка переносит A вместе со всей строкой, номер никогда не пересчитывается.
Отсутствие ledger-row означает «ещё не запланировано». `reserved` — единственное промежуточное состояние: оно нужно только для повторяемого номера и crash recovery; `written` не сохраняется, а факт записи доказывает свежий полный readback Google.

Настоящая неопознанная физическая строка ещё не имеет `call_key`, поэтому в call-ledger её записать нельзя. Для неё в той же SQLite добавляется второй маленький технический реестр без бизнес-текста:

```text
google_publication_incidents
  destination_id              TEXT
  incident_id                 TEXT
  observed_row_sha256         TEXT
  reason_code                 TEXT
  candidate_call_key_hashes   JSON|null
  status                      open|resolved
  first_seen_at               UTC timestamp
  last_seen_at                UTC timestamp
  resolution_by               TEXT|null
  resolution_at               UTC timestamp|null
  resolution_reason_code      TEXT|null
  PRIMARY KEY(destination_id, incident_id)
```

`incident_id` детерминированно строится из destination, observed row hash, reason code и номера occurrence. Повторный запуск той же нерешённой причины обновляет `last_seen_at`, но не сбрасывает `first_seen_at`. Если уже разрешённая проблема появляется снова, старое решение не стирается: создаётся следующий occurrence с новым `first_seen_at`, немедленным alert и собственной резолюцией. Телефон, конспект, transcript и другие ПДн в incidents не попадают. Открытый identity incident блокирует write/sort/layout; `identity_quarantine` в call-ledger применяется только когда известны один или несколько candidate call keys.

Почему нужна таблица, а не только полный diff:

- после изменения transcript текущий row hash меняется с обеих сторон;
- A:P не содержит call key;
- бизнес-сигнатура может быть неоднозначной;
- сбой возможен между Google write и SQLite commit;
- старый `sync_status=done` не хранит, какая версия была подтверждена.

Полная сверка всё равно выполняется каждый запуск: ledger помнит идентичность/последний verified hash, но не объявляет Google корректным без свежего readback.

Постоянный hidden call-key column в Google сознательно не используется: рабочий контракт требует ровно 16 бизнес-колонок и пустые Q:Y; скрытый технический ID всё равно попадает в экспорт, может быть удалён/изменён пользователем и становится вторым авторитетным реестром вне owner-only SQLite. Временная Q допустима только для атомарной сортировки и очищается. Поэтому выбран гибрид `локальный ledger + полный live readback`, а не скрытая колонка.

## 6. Версии и fingerprint

Для каждого готового звонка детерминированно строятся:

- `analysis_result_sha256`: SHA-256 канонического JSON всех выходов Analyse, используемых проекцией: `history_summary`, структурированных полей, claims, role/review flags, normalized facts и их значений;
- `source_fingerprint`: tenant/provider/call key, transcript SHA, `canonical_dialogue_sha256`, `analysis_input_sha256`, `analysis_result_sha256`, schema/prompt versions, `role_guard_sha256`, claim/evidence version, normalizer engine/ruleset/tenant/timezone versions, а также SHA/version фактически прочитанного справочника менеджеров;
- `projection_version`: версия ровно 16 колонок и layout-контракта;
- `desired_row_sha256`: SHA-256 канонического JSON точных A:P после безопасного форматирования.

Любое изменение источника/версии/значения делает строку stale независимо от `sync_status=done`.

## 7. Ровно 16 видимых колонок

1. №
2. Дата и время МСК
3. ФИО менеджера
4. Направление
5. Длительность
6. Категория
7. Телефон
8. Нужна проверка
9. Тема
10. Конспект разговора
11. Результат
12. Возражение/причина
13. Следующий шаг
14. Срок
15. Что проверить РОПу
16. Полная расшифровка

Правила:

- A:P принадлежат автоматизации; ручные заметки туда не добавляются.
- A:P защищаются как output-only range: пользователи читают, фильтруют и раскрывают ячейки, но не используют значения как ручной источник истины. Случайная правка не принимается обратно в SQLite/KB. Будущий ручной review требует отдельного контролируемого ввода с call key/evidence и не входит в это ТЗ.
- Полная расшифровка остаётся в P, но визуально обрезана `CLIP`; клик/раскрытие показывает полный текст.
- Высота строки рассчитывается после финальной сортировки только из J по формуле §3.
- Trusted roles отображаются `Менеджер/Клиент`; untrusted — `Спикер A/Спикер B`, никогда tentative Manager/Client.
- Колонка H различает `Нет`, `Да: <человекочитаемая причина>`, `Legacy: не проверено`.
- O показывает причины/таймкоды claims без raw технических кодов и ПДн.
- Summary не содержит дату/менеджера: они уже в B/C.
- Пустой transcript, invalid high-risk projection или ambiguous identity не публикуются молча.

### 7.1. Единственная разрешённая проекция A:P

| Колонка | Разрешённый источник | Обязательный guard | Fallback |
|---|---|---|---|
| A № | `google_publication_ledger.display_number` | уникальный положительный номер в пределах destination | нет; ошибка назначения блокирует эту строку |
| B Дата/время МСК | `CallRecord.started_at` через общий formatter ТЗ-04 | исходное naive-время трактуется как UTC, перевод ровно один раз | нет |
| C ФИО менеджера | `CallRecord.manager_name` через точное `mapping[manager_name]` | SHA/version справочника входит в fingerprint; fuzzy запрещён | человекочитаемое исходное имя либо `Не определён` |
| D Направление | `CallRecord.direction` | закрытое отображение: `incoming/inbound→Входящий`, `outgoing/outbound→Исходящий` | `Не определено` |
| E Длительность | `CallRecord.duration_sec` | finite, `>=0`, округление один раз до секунды | data error, строка не публикуется |
| F Категория | `analysis_json.quality_flags.call_type` | закрытое отображение: `sales_call→Продажа`, `service_call/existing_client_progress→Сервис`, остальное→`Не определено` | `Не определено` |
| G Телефон | `CallRecord.phone` | только строковое отображение Mango metadata, без вывода в журналы | пустая строка |
| H Нужна проверка | `analysis_json.needs_review` + `quality_flags.review_reasons` + состояние legacy | только current reason codes ТЗ-02/03/04 | `Legacy: не проверено` |
| I Тема | valid current claims по `structured_fields.interests.{products,subjects,format,exam_targets}` и их current `normalized_facts`; для untrusted — только закрытая `quality_flags.neutral_topic` ТЗ-02 | каждый high-risk элемент имеет valid claim; stale normalization запрещена | пусто |
| J Конспект | `analysis_json.history_summary` current-схемы v3 | тот же `analysis_input_sha256`; все high-risk утверждения имеют valid claim/summary refs; role guard применён | для v1/v2 — собственный текущий конспект с H=`Legacy: не проверено`; при нарушении v3 — безопасный детерминированный конспект либо пусто+review |
| K Результат | `analysis_json.structured_fields.result.status` + `.detail` | status отображается закрытым словарём ТЗ-03, detail добавляется только со своим current valid explicit claim; role-dependent status требует trusted role | пусто |
| L Возражение/причина | `analysis_json.structured_fields.objections[]` | отдельный current valid explicit claim на каждый item; trusted role для атрибуции | пусто |
| M Следующий шаг | `analysis_json.structured_fields.next_step.action` | current valid explicit claim + `role_attribution.trusted=true` | пусто |
| N Срок | current `normalized_fact` для `structured_fields.next_step.due`, ссылающийся на valid due-claim | trusted role, current timezone/ruleset/claim versions; raw неоднозначность не отображается как точная дата | пусто |
| O Что проверить РОПу | человекочитаемое отображение `review_reasons` и `[таймкод]` связанных invalid/missing claims | закрытое отображение без raw JSON-кодов, transcript и ПДн | пусто только когда H=`Нет` |
| P Полная расшифровка | полный детерминированный render канонического DialogueInput ТЗ-01 до prompt-truncation | исходный DialogueInput имеет `canonical_dialogue_sha256`; trusted roles→Менеджер/Клиент, untrusted→Спикер A/B | нет; пустой/повреждённый transcript = data error |

Нормализованное значение используется только если оно current, ссылается на тот же valid claim и имеет ожидаемые tenant/engine/ruleset/timezone versions. Иначе raw остаётся только там, где он сам разрешён контрактом; для суммы, даты, срока, оплаты, обещания и иных high-risk полей публикуется пусто+review. Для legacy v1/v2 K:N пусты до повторного Analyse, если Дмитрий отдельно письменно не утвердит иной временный режим; legacy-данные не используются для автоматических действий и Customer Timeline.

Закрытое отображение K: `information_only→Информация предоставлена`, `follow_up_agreed→Согласован повторный контакт`, `appointment_agreed→Согласована встреча/занятие`, `offer_sent→Материалы/предложение отправлены`, `sale_agreed→Согласована покупка`, `payment_confirmed→Оплата подтверждена`, `refusal→Отказ`, `no_decision→Решение не принято`, `non_conversation→Нет содержательного диалога`. `payment_confirmed` запрещено получать из намерения купить/обсуждения суммы; нужна отдельная явная claim-ссылка о состоявшейся оплате. Неизвестный status не отображается и создаёт review.

## 8. Алгоритм одного запуска

1. Получить `flock`; если занят — штатно завершить без второго writer.
2. Открыть SQLite в WAL/read-only для чтения, `busy_timeout=30000`, проверить `quick_check=ok`.
3. Построить два разных набора. `identity_scope` включает все call keys из ledger/reservations и соответствующие CallRecord независимо от текущего `analysis_status`, а также все новые `analysis_status=done`. `writable_desired_scope` включает только звонки с завершённым current Analyse и валидным source fingerprint. Эти наборы нельзя смешивать.
4. Для `writable_desired_scope` построить desired B:P/fingerprints; A взять из существующего ledger, а для новых строк пока оставить неназначенным. Для исторического звонка в `analysis_status=pending|in_progress|failed` прежняя физическая строка остаётся связанной по `last_verified_row_sha256`, получает канонический Q из CallRecord, но не меняется и не получает новый verified. Когда v3 Analyse станет `done`, fingerprint делает её stale и она обновляется обычным путём.
5. Полностью прочитать Google A:Y и проверить точные headers, непрерывность, форматы, дубли и неожиданные данные. Непустая Q:Y считается следом незавершённой/чужой операции: до identity gate новые записи запрещены; после однозначного сопоставления она восстанавливается только тем же атомарным алгоритмом ниже.
6. Однозначно сопоставить:
   - сначала текущий desired row hash для уже пронумерованных calls;
   - затем `planned_row_sha256` активной reservation;
   - затем `last_verified_row_sha256` ledger;
   - только для первичного bootstrap — точные время/телефон/duration/transcript SHA с legacy-допусками.
   До сопоставления строится обратный индекс `row_hash → call_keys/physical rows`. Один hash с несколькими call keys или физическими строками не связывается по первому совпадению: все участники collision-group получают `ambiguous_row_hash_collision` и `identity_quarantine`.
7. Любая уже существующая физическая A:P строка без однозначного call key (ambiguous/cross-match/unidentified/identity quarantine) блокирует весь Google write/sort/layout этого запуска. Publisher upsert-ит соответствующий incident, сохраняя исходный `first_seen_at`, и завершает работу. Missing desired call, которого ещё нет в Google, не считается неопознанной физической строкой и может быть добавлен после прохождения global gate. Никакого best-effort порядка по видимой B нет.
8. После global identity gate выбрать не более 25 missing/stale calls. Короткой `BEGIN IMMEDIATE` создать/обновить reservations: переиспользовать прежний `display_number`, а новым calls назначить `max(display_number в ledger и однозначно распознанных A)+1` в стабильном порядке call key. Записать source fingerprint и полный `planned_row_sha256`; при изменившемся source обновить plan, но не номер. `sync_status` не менять.
9. После identity gate подготовить один атомарный Google `spreadsheets.batchUpdate` с упорядоченными requests: append/update не более 25 зарезервированных A:P; запись Q для **каждой** однозначно сопоставленной physical row; `sortRange` A:Q DESC; очистка всей Q; явный J=`WRAP/TOP`, P=`CLIP/TOP`, ширина J и row heights from J. Скалярный ключ Q строго равен `unix_seconds(started_at_utc) * 1_000_000 + display_number`; перед write доказать `0 < display_number < 1_000_000`, результат — точное целое `<2^53`. Пропуски/stale приоритетнее новых; не-due row не занимает слот. Google применяет весь batch атомарно; отдельные append/Q/sort/clear/layout calls запрещены.
10. Если ответ batchUpdate потерян/timeout, не повторять запись вслепую: выполнить полный readback A:Y. При непустой Q или частично неизвестном результате сначала снова пройти identity gate; затем повторить единый идемпотентный batch из reservation. До этого ledger не становится verified.
11. Повторно прочитать весь A:Y и доказать:
    - desired и Google multiset совпадают для writable/verified scope, а временно pending/in-progress historical rows по-прежнему однозначно связаны и побайтно не изменились;
    - нет пропусков/дублей/cross-row;
    - J/P принадлежат одному call key;
    - сортировка newest-first точна до секунд/tie-breaker;
    - Q:Y пусты;
    - P непустая, CLIP/TOP; высота соответствует J.
12. Только после readback одной короткой SQLite-транзакцией перевести reservation в `verified`, перенести `planned_row_sha256` в `last_verified_row_sha256`, очистить planned/reserved fields и поставить совместимый `sync_status=done`. Между ledger/sync записями commit невозможен. Перед commit заново прочитать CallRecord и справочник, пересчитать `analysis_result_sha256`, весь `source_fingerprint` и `desired_row_sha256`; оба итоговых хэша должны совпасть с reservation и readback. Любое расхождение откатывает транзакцию, оставляет call `reserved` и перепланируется в следующем запуске.

## 9. Сбои и идемпотентность

- Сбой до reservation: локальный state не меняется.
- Сбой после reservation до Google write: следующий запуск использует тот же `display_number` и `planned_row_sha256`.
- Сбой после Google write до verified-commit: следующий полный read находит строку по `planned_row_sha256` и завершает reservation без дубля или перенумерации.
- Частичный batch: следующий запуск сверяет весь лист и продолжает только отсутствующее/stale.
- Ошибка readback: ledger не получает verified, `sync_status` не подтверждается.
- Ручное изменение A:P при потерянной/неоднозначной identity → `identity_quarantine`, без автоматического угадывания.
- Повтор без изменений: `0` Google write и те же hashes.
- Второй процесс не получает lock.
- Publisher не очищает lease/claims ASR и не меняет analysis_json.

Классы ошибок:

- `identity_quarantine`: существующая физическая строка не имеет единственного доказанного call key. Она блокирует весь write/sort/layout до owner-resolution, потому что publisher не знает её канонический Q и не имеет права переставлять её наугад;
- `data_quarantine`: call key доказан, но желаемая проекция этой строки невалидна либо локальная API 400 повторилась после трёх изолированных попыток. Для существующей строки сохраняется последний verified A:P и она сортируется по каноническому DB-key; новая невалидная строка пока не добавляется. Состояние видно в health, но остальные однозначные строки продолжают публиковаться;
- 429, 5xx, timeout/network → transient destination error, учитывать `Retry-After`/exponential backoff с jitter, не quarantine row и не увеличивать её data-attempts;
- 401/403, неверный sheet/tab/header contract → fatal destination stop и alert, без пометки отдельных calls как плохих.

Один не-due/error call не занимает batch slot. Повторяющаяся destination error видима в health/status, но не меняет verified hashes.

Выход из identity/data quarantine или открытого incident существует только через owner-only admin dry-run: отчёт показывает call key/hash/причину без текстов и ПДн. Для identity-проблемы предложенное сопоставление строится теми же bootstrap-инвариантами; для data-проблемы показывается исправленная source-проекция. Дмитрий или назначенный владелец письменно выбирает: восстановить desired A:P, перенести нужную информацию в будущий контролируемый review-контур либо подтвердить уникальное rematch. Принимать текущее ручное значение Google как новый бизнес-факт запрещено: Google — витрина, не источник Analyse/KB. После readback ledger получает resolution_by/at/reason и новый verified hash, а incident — `resolved`; автоматического self-heal неоднозначности нет.

Так как одна `identity_quarantine` physical row блокирует весь writer, до rollout обязательно задаются непустые `publisher_incident_owner`, `publisher_alert_sink` и `quarantine_resolution_sla_sec`; без них publisher не включается. Alert sink должен быть уже существующим owner-only каналом эксплуатационных тревог и подтверждать доставку структурированного кода без ПДн; конкретный transport выбирается при внедрении и входит в config fingerprint. Для текущего бизнеса рекомендуемый начальный SLA — 30 минут: alert создаётся в первом же запуске, при превышении SLA состояние явно красное, а не «publisher healthy». Повторное появление разрешённого incident немедленно открывает новый occurrence и alert. `data_quarantine` не блокирует другие строки, но не считается verified и имеет тот же явный owner/SLA. Изменение SLA требует business-sign-off.

## 10. Bootstrap существующего листа

До первой записи выполнить dry-run всей таблицы:

- построить mapping существующих строк к call keys;
- проверить A как положительные уникальные значения и сохранить их как `display_number`; значения не перенумеровывать из-за сортировки;
- отдельно показать uniquely matched, legacy matched, missing, extra, duplicate, ambiguous;
- для extra/unidentified/ambiguous physical rows создать incidents с устойчивым `first_seen_at`; при их наличии bootstrap остаётся read-only для Google;
- проверить известные восстановленные id 44/67/68 как регрессию;
- найти все stale summaries, включая исторические 53 из аудита;
- не записывать ledger для неоднозначных строк;
- не исправлять Google до письменного одобрения отчёта Дмитрием.

Первоначальный ledger заполняется только после точного readback текущих строк; он не «узаконивает» ошибочную проекцию как правильную, а хранит её verified row hash для безопасного будущего обновления.

### 10.1. Граница первого выпуска и корпус M4

Первый минимальный выпуск обслуживает текущую M1 working SQLite и существующий рабочий лист. Прямая публикация примерно 65 тысяч исторических звонков M4 в тот же Google-лист **не входит**: она требует отдельного ТЗ по размеру витрины, стоимости полного readback, retention и пользовательскому назначению. M4 сначала передаётся как immutable read-only snapshot для канонического архива/Customer Timeline, а не автоматически раздувает оперативный лист.

Это сознательное упрощение устраняет ложное обещание линейного bootstrap: текущий алгоритм каждого continuous batch один раз пишет полный Q/sort/layout для нынешнего листа, но не повторяет такую глобальную работу десятки/сотни раз на 65 тысячах строк. Будущее масштабное ТЗ обязано либо делать один финальный глобальный Q/sort/layout после targeted writes, либо доказать другой алгоритм с суммарной работой `O(N)` по cells/bytes, crash-resume и полным финальным readback. До такого GO режим `bootstrap_session` отсутствует в production-коде.

Перед записью проверяется фактическая длина каждой ячейки. Проектный fail-closed предел — не более 50 000 Unicode-символов в одной ячейке; [официальная справка Google Drive/Sheets](https://support.google.com/drive/answer/37603) указывает 50 000 символов как границу при преобразовании в Sheets. Текущий проверенный максимум полной расшифровки — 25 390 символов, то есть текущего блокера нет. Значение сверх проектного предела не обрезается и не заменяется молча: call получает `data_quarantine` до отдельного утверждённого контракта хранения длинного transcript.

## 11. Тесты

1. Legacy SQLite мигрируется без изменения call records.
2. Stable call key не зависит от local id/filename/row.
3. Изменение Analyse/input/result/ruleset/timezone/справочника менеджеров/projection меняет fingerprint.
4. Неизменный запуск делает zero writes.
5. Stale `sync_status=done` обновляет ту же строку.
6. Изменение transcript также обновляет правильный call key через ledger.
7. Duplicate/ambiguous identity блокируется до Google write.
8. Crash до/после write/readback/ledger commit не создаёт дубль.
9. Два параллельных запуска дают одного writer.
10. Сортировка переносит A:P целиком, сохраняет multiset и newest-first.
   Тест включает запрет любых Google writes/sort при хотя бы одной неопознанной существующей physical row.
11. Exact 16 headers; Q:Y empty.
12. Duration `2 мин 23 с`, UTC→MSK once.
13. Trusted/untrusted speaker labels и review reason.
14. Formula injection во всех текстовых колонках.
15. Full transcript непустой, P CLIP/TOP.
16. Row height вычислена только из J по точной формуле; длинная P её не увеличивает.
   Readback также проверяет фактическую ширину J; при другой ширине сначала восстанавливает версионированную `SUMMARY_COLUMN_WIDTH_PX`.
17. 53 известных stale summaries обнаруживаются dry-run.
18. Изменение source, результата Analyse или справочника менеджеров между Google write и DB commit не получает false verified; перед commit совпадают заново рассчитанные source и desired-row hashes.
19. Два call key с одинаковым desired row hash не подтверждаются «первым совпадением» и оба попадают в `identity_quarantine`.
20. Owner-approved quarantine/incident resolution оставляет audit fields и только после readback возвращает call в verified/incident в resolved.
   Отдельный тест подтверждает немедленный alert и unhealthy-state после configured SLA.
21. Crash между ledger/sync writes откатывает обе записи одной транзакцией.
22. 429/5xx/network не создают row quarantine и после backoff не голодают другие due rows; isolated deterministic row error после лимита создаёт только `data_quarantine`; 401/403 останавливает destination.
23. Bootstrap сохраняет существующие уникальные A, новые calls получают уникальные возрастающие `display_number`; сортировка не меняет их и не вызывает массовый stale.
24. Crash до/после reservation, Google-write, readback и final commit сохраняет тот же номер, не создаёт дубль и переводит `reserved→verified` только после exact readback.
25. Неопознанная физическая строка создаёт устойчивый incident: повторный запуск сохраняет `first_seen_at`, обновляет `last_seen_at`, SLA не начинается заново; resolution остаётся в audit trail.
26. `identity_quarantine`/open identity incident запрещает все writes/sort/layout; `data_quarantine` с доказанным call key не блокирует другие due rows и не считается verified.
27. Каждая из 16 колонок использует только path/guard/fallback из §7.1; stale normalized fact, invalid claim и legacy K:N не проходят.
28. Append/update, полный Q, sort, clear и layout находятся в одном batchUpdate; injected failure любой request оставляет лист без частичного изменения.
29. Timeout с неизвестным результатом вызывает readback, а не слепой append; обнаруженная Q восстанавливается/очищается только после identity gate, повтор не создаёт дубль.
30. Формула Q сортирует одинаковые секунды по стабильному `display_number`, проверяет `<1_000_000` и `<2^53`; выход за пределы fail-closed требует новой projection version.
31. Попытка включить несуществующий M4/65k bootstrap-режим в первом выпуске fail-closed; прямой массовый экспорт требует отдельной утверждённой спецификации.
32. Continuous batch остаётся в измеренном payload/quota, учитывает `Retry-After`, применяет backoff и возобновляется без дублей.
33. Transcript сверх лимита Google не обрезается, не получает verified и не блокирует публикацию других однозначных calls.
34. Полный переход `legacy verified → analysis pending/in_progress → v3 done → Google readback → new verified` сохраняет identity старой строки и не блокирует одновременную публикацию нового звонка.
35. `destination_id` различает вкладки, tenant и environment; изменение формулы без миграции fail-closed.
36. Одинаковый resolved incident, появившийся повторно, создаёт новый occurrence и alert, не стирая прошлую резолюцию.

## 12. Rollout и GO/STOP

1. Unit/integration tests с fake Sheets.
2. Dry-run на копии SQLite и export/readback копии листа.
3. Полный bootstrap report production только чтением.
4. Shadow нового publisher рядом со старым writer, без write. ТЗ-01…ТЗ-04 в это время также остаются shadow и не отдают новые payload старому writer.
5. Production-cutover выполняется совместимым комплектом ТЗ-01…ТЗ-05: остановить только старый writer, не Capture/ASR/Resolve; включить новый Analyse+publisher, канарейка 25 строк, readback. Допустим предварительный compatibility-cutover ТЗ-05 для старой схемы, если он отдельно прошёл все gates; обратный период «новый Analyse + старый writer» запрещён.
6. 24 часа и не менее 200 изменений, затем расширение.

Два GO нельзя смешивать:

- `transport_go` доказывает правильную строку, версию, сортировку, layout и readback. Compatibility-режим с J=`Legacy: не проверено` может получить только этот статус;
- `content_quality_go` доказывает, что целевой исторический scope повторно прошёл Analyse v3, J собран детерминированно из valid claims, legacy/stale summaries в этом scope равны нулю и бизнес-выборка принята. Только этот статус означает, что массовая проблема конспектов исправлена и данные допустимы для Customer Timeline/KB.

`transport_go`:

- `100%` существующих физических строк однозначно связаны с call key; `identity_quarantine=0`, открытых identity incidents=`0`;
- `0` duplicate/cross-call/missing verified rows;
- `0` stale verified projections;
- `data_quarantine` явно видна, не считается verified и не превышает owner SLA; на первичном GO её значение равно `0`;
- повтор без изменений = `0` writes;
- `100%` writes имеют exact readback до ledger verified;
- newest-first, МСК, duration, 16 headers, J/P/layout выполняются;
- производительность publisher минимум на 20% выше устойчивой скорости новых Analyse и backlog не растёт;
- Capture/ASR/Resolve/Analyse не прерывались.

`content_quality_go` дополнительно:

- в целевом scope `analysis_schema_version=3`, `history_summary_meta.contract_version` current и H не содержит `Legacy`;
- `100%` high-risk фраз J воспроизводятся из current valid claims либо удалены с review;
- исторические 53 stale own summaries обновлены по current fingerprint;
- выполнены бизнес-гейты ТЗ-02/03/04 по ролям, evidence, нормализации и ручной аудиовыборке;
- только после этого scope разрешён для Customer Timeline/KB.

STOP:

- хотя бы один summary/transcript оказался у другого call key;
- потеря/дубль/неоднозначное автоисправление;
- хотя бы одна unresolved `identity_quarantine` или открытый identity incident;
- false verified после сбоя;
- два writer одновременно;
- Q/технический key остался видимым;
- row height снова определяется P;
- publisher не успевает за Analyse или тормозит SQLite.

## 13. Откат

- Остановить только новый publisher после текущего запроса; ASR не трогать.
- Ledger остаётся audit trail и не удаляется.
- Новые Analyse payload остаются локально pending; при необходимости временно останавливается только Analyse→Google, а Capture/ASR/Resolve продолжают работать.
- Старый writer отдельно не возвращается к новым payload: он уже доказанно нарушает role guard. Вернуть можно только весь прежний совместимый комплект после явного schema-gate либо выполнить fix-forward нового publisher; одновременно два writer запрещены.
- Ledger поддерживает текущее verified-состояние, но не хранит произвольную историю N-1. Автоматический row-level rollback к прошлой проекции не обещается: безопасный путь — остановить writer, сохранить текущий readback и выполнить fix-forward/republish из доказанного источника. Нативная Google Version History может использоваться только по отдельному owner-решению после сравнения с SQLite; live SQLite целиком не откатывается.

## 14. Что не делать

- не оставлять бизнес-логику в automation prompt;
- не использовать номер строки/видимый №/local id как identity;
- не считать `sync_status=done` доказательством актуальности;
- не копировать summary отдельно от всей A:P строки;
- не писать source_call_id/SHA в видимые колонки;
- не использовать `autoResizeDimensions`;
- не создавать вторую call DB или второй Google writer;
- не сбрасывать всю историю до bootstrap dry-run.

## 15. Журнал независимых аудитов Claude CLI

Максимум пять раундов; финальный GO требует отсутствия новых P0/P1.

### Раунд 1 — идентичность и необходимость ledger

Claude CLI: `REVISE`.

Принято:

- обратный индекс row hash обязателен; коллизии разных call keys/строк не разрешаются первым совпадением и уходят в quarantine;
- добавлен owner-approved путь восстановления quarantine с audit полями и обязательным readback;
- явно рассмотрен и отклонён постоянный hidden key в Google: он нарушает 16-column/Q:Y-empty контракт и не является надёжным owner-only источником истины.

### Раунд 2 — crash recovery и классы ошибок

Claude CLI: `REVISE`, новых P0 нет.

Принято:

- ledger verified и совместимый `sync_status` обновляются одной SQLite-транзакцией;
- ошибки разделены на identity/data, isolated row, transient destination и fatal destination; retry/backoff не блокирует остальную очередь.

Уточнение после спора:

- предложение переводить любую строку в quarantine после пяти 429/5xx не принято: это сбой Google/destination, а не дефект row. Quarantine применяется только к доказанной identity/data/isolated-row проблеме; 401/403 останавливают весь publisher.

### Раунд 3 — ручные правки и геометрия конспекта

Claude CLI: `REVISE`.

Принято:

- ширина J стала частью layout/projection contract и проверяется до формулы высоты;
- A:P явно защищены как output-only, accidental edit имеет owner-resolution, а future manual review вынесен в отдельный доказательный контур.

Отклонено:

- принимать ручное содержимое Google как override бизнес-факта. Это превратило бы витрину в неподтверждённый источник KB и обошло бы ТЗ-03. Предложенный Claude anchor с provider call key также нельзя вычислить из A:P без скрытого технического ID, который сознательно запрещён контрактом 16 колонок/Q:Y empty.

### Раунд 4 — минимальность state machine и честный откат

Claude CLI: `REVISE`, новых P0 нет.

Принято:

- ledger status сокращён до реально используемых `verified/error/quarantine`; отсутствие строки означает pending, восстановление опирается на full readback;
- снято неподдерживаемое обещание произвольного N-1 rollback. Журнал хранит текущее verified-состояние, rollback — stop/fix-forward; Google Version History используется только после отдельной сверки владельцем.

### Раунд 5 — сортировка при неопознанной строке

Claude CLI: первоначально `STOP` из-за неопределённого Q для quarantined physical row.

Проблема закрыта более строгим вариантом:

- если хотя бы одна существующая A:P строка не имеет однозначного call key, весь Google write/sort/layout запрещён до owner-resolution;
- Q всегда строится только из канонического DB после полного identity gate; видимая B никогда не становится fallback sort key.

Предложенный Claude best-effort sort по отображаемой B отклонён: исторически minute-only/UTC строки уже создавали канонические инверсии. Исправленный контракт сохраняет текущий fail-closed принцип живой автоматизации. После устранения найденного разрыва других P0/P1 в заключении не указано; пост-аудитное состояние считается условным `GO` без шестого раунда.

### Перекрёстный аудит пяти ТЗ после раунда 5

Это не шестой раунд Claude, а отдельная проверка стыков несколькими агентами Codex. Она уточнила контракт без изменения честного статуса Claude:

- старое сокращение статуса до одного `quarantine` заменено на `reserved`, `identity_quarantine`, `data_quarantine` и отдельный incidents registry: reservation обеспечивает crash recovery номера, а только потерянная identity блокирует весь лист;
- добавлены `analysis_result_sha256` и SHA/version справочника менеджеров, а перед ledger-commit повторно проверяются source и row hashes;
- закреплена точная таблица source/guard/fallback для всех A:P;
- видимый № стал стабильным `display_number`, не зависящим от физической строки;
- production-cutover согласован с ТЗ-01…ТЗ-04: новый Analyse никогда не обслуживается старым небезопасным writer.

Claude не видел эти post-fix дополнения повторно. Честный статус нынешней редакции: `условный post-fix GO по пяти раундам Claude + отдельный перекрёстный аудит Codex`; строка раунда 4 про упрощённый `verified/error/quarantine` сохранена только как исторический журнал и не описывает финальную state machine.
