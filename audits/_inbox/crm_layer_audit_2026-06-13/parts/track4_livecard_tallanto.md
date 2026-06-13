# Аудит: живая карточка (live_card) + слой Tallanto
**Дата:** 2026-06-13  
**Ревизор:** субагент read-only  
**Статус:** проверено по коду; боевой коннектор не вызывался

---

## A. Карта модулей live_card + Tallanto

| Модуль | Файл | Назначение | Вход | Кто запускает |
|---|---|---|---|---|
| `tallanto_api.py` | `amocrm_runtime/` | HTTP-клиент к Tallanto REST API; поиск контакта по телефону; загрузка связанных сущностей (Opportunity, Request, Finances, Abonements, ClassRelations, Classes) | `phone / contact_id`, конфиг | `tallanto_context.py` |
| `tallanto_context.py` | `amocrm_runtime/` | Сборка живого контекста: вызывает API, строит компактные срезы, делегирует `build_tallanto_live_card` | `phone, tallanto_id, tallanto_match_status, active_brand` | `deal_dossier.py` строка 215 |
| `build_tallanto_live_card` | `tallanto_context.py` строка 242 | Из `live_contexts` строит структуру `live_card_v1`: `payments`, `balance`, `schedule`, `enrollment`, TTL, provenance | `contexts` (список), `active_brand`, `matched_via` | `build_live_tallanto_context` строка 226 |
| `deal_dossier.py` | `amocrm_runtime/` | Собирает полное досье сделки; вызывает `build_live_tallanto_context` (строка 215); помещает результат в `tallanto_live` (строка 378) | `phone_context`, `contact`, `lead`, `notes`, `tasks` | `deals.py` строка 857 |
| `deal_llm.py` | `amocrm_runtime/` | Компактирует досье для промпта; включает `tallanto_live` целиком в JSON (строка 218) — без дополнительной обрезки | дossier dict | вызывается из `deals.py` строка 873 |
| `tallanto_matching.py` | `amocrm_runtime/` | Матчинг контакта по телефону из заранее загруженного списка; вычисляет `IdentityMatchResult` (score, ambiguous) | список контактов + `call_phone` | вызывается из `deals.py` (отдельный путь, не из живой карточки) |
| `tallanto_deal_ranking.py` | `amocrm_runtime/` | Ранжирование возможностей (Opportunity) по времени, статусу, менеджеру | список opportunities | вызывается из `deals.py` |
| `tallanto_premature_close.py` | `amocrm_runtime/` | Детерминированная оценка риска «преждевременного закрытия» по сигнальному набору; данных Tallanto напрямую не читает | `PrematureCloseSignals` | `deals.py` |
| `tallanto_export.py` | `amocrm_runtime/` | Утилита выгрузки схемы и снимков модулей; в боевом пути NOT используется | `TallantoApiClient` | CLI/ручной запуск |
| `phone_context.py` | `amocrm_runtime/` | Кеш CSV-файлов экспорта; выдаёт `PhoneContext` с `tallanto_id` и `tallanto_match_status` из CSV | `phone` | `deals.py` |

---

## B. Поток данных Tallanto → живая карточка

```
deals.py → build_deal_dossier(phone_context, ...) [строка 857]
  └─ phone_context.tallanto_id / .tallanto_match_status  ← из CSV (phone_context.py)
     └─ build_live_tallanto_context(phone, tallanto_id, tallanto_match_status)
          [tallanto_context.py строка 73; active_brand НЕ ПЕРЕДАЁТСЯ из deals.py — см. раздел D]
          │
          ├─ если tallanto_id + статус в {exact_phone_single, manual_confirmed, id_confirmed}:
          │    client.build_contact_context_by_contact_id(tallanto_id)
          │    ├─ contact_by_id()  → 1 HTTP-запрос
          │    ├─ opportunities_by_contact()  → iter_entry_list (paginated)
          │    ├─ requests_by_contact()       → iter_entry_list (paginated)
          │    ├─ finances_by_contact()       → iter_entry_list (paginated)
          │    ├─ course_relations_by_contact() → iter_entry_list (paginated)
          │    ├─ class_relations_by_contact()  → iter_entry_list (paginated)
          │    ├─ abonements_by_contact()       → iter_entry_list (paginated)
          │    └─ classes_by_ids(class_ids)    → N × get_entry_by_id (ЦИКЛ)
          │
          └─ иначе (поиск по телефону):
               client.build_contact_context(phone, max_contacts=5)
               ├─ search_contacts_by_phone()  → до 5×8 HTTP-запросов (цикл поля×кандидат)
               └─ для каждого найденного контакта: те же 7 вызовов выше
```

Из `live_contexts` берутся только: `contact`, `finances`, `abonements`, `class_relations`, `classes`.  
`opportunities` и `requests` попадают в `compact_contexts` (для API-ответа), но **НЕ** в `live_card`.

`build_tallanto_live_card` строит `live_card_v1`:
- `payments` ← `finances` (сортировка по `date_entered`, лимит 5)
- `balance` ← `abonements` (фильтр `_record_is_active`, поле `num_visit_left`, лимит 5)
- `schedule` + `enrollment` ← `classes` (фильтр `_record_is_active`, лимит 8)

`live_card` целиком включается в промпт через `deal_llm.py` строка 218 → JSON без дополнительной обрезки.

---

## C. Неэффективность

### C1. N+1 запросов: `classes_by_ids` в цикле
**Файл:** `tallanto_api.py` строки 521–544  
**Суть:** метод итерирует по `class_ids` и вызывает `get_entry_by_id(module="most_class", ...)` **на каждый ID отдельно**. При типичном клиенте с 5–10 группами — 5–10 последовательных HTTP-запросов только за расписанием.  
```python
for class_id in class_ids:          # строка 529
    ...
    payload = self.get_entry_by_id(module="most_class", entry_id=value)  # строка 535
```
Нет batch-эндпойнта в REST Tallanto — это ограничение API, но можно добавить параллельность или заранее ограничить число `class_ids` до лимита карточки (8).

### C2. Поиск контакта по телефону: двойной вложенный цикл
**Файл:** `tallanto_api.py` строки 409–427  
**Суть:** `search_contacts_by_phone` гоняет цикл `поле (5) × кандидат (~8)` = до 40 HTTP-запросов ради одного контакта. Каждый запрос — отдельный `get_entry_by_fields`. Если `tallanto_id` не передан или невалиден — этот путь активируется по умолчанию.  
```python
for field_name in self.CONTACT_PHONE_FIELDS:   # строка 409  (5 полей)
    for candidate in candidates:                # строка 410  (~8 вариантов)
        payload = self.get_entry_by_fields(...)  # строка 412
```
При успешном совпадении цикл не прерывается немедленно (проверяется только `>= max_records`), поэтому возможны лишние запросы даже после нахождения контакта.

### C3. Повторное создание `TallantoApiClient` на каждый вызов
**Файл:** `tallanto_context.py` строка 90  
```python
client = TallantoApiClient(build_tallanto_api_config())
```
Клиент не кешируется: при каждом входящем сообщении создаётся новый объект + вызов `build_tallanto_api_config()` (чтение настроек, нормализация URL). Невысокая цена, но лишняя работа в пилотном контуре где на сообщение один вызов.

### C4. `iter_entry_list` всегда загружает до `max_related_records=100`
**Файл:** `tallanto_context.py` строки 73–119; `tallanto_api.py` строки 363–397  
`build_live_tallanto_context` передаёт `max_related_records=40` по умолчанию, что разумно. Но `build_contact_context` и `build_contact_context_by_contact_id` вызывают все 7 методов с тем же лимитом, включая `opportunities` и `requests` — которые в `live_card` **вообще не используются** (только в `compact_contexts`). Это бесполезный трафик при каждом входящем сообщении.

### C5. `tallanto_export.py`: `discover_tallanto_schema` в цикле по модулям
**Файл:** `tallanto_export.py` строки 43–45  
```python
field_catalog = {
    module: client.list_possible_fields(module)
    for module in selected_modules
}
```
9 последовательных запросов. Используется только в CLI/ручном режиме, в боевом пути не задействован — низкий приоритет.

---

## D. Обрезка и порча при сборке живой карточки

### D1. КРИТИЧНО: `active_brand` не передаётся из `deals.py` в `build_deal_dossier`
**Файл:** `deal_dossier.py` строки 215–219  
```python
tallanto_live = build_live_tallanto_context(
    phone=phone_context.phone,
    tallanto_id=phone_context.tallanto_id,
    tallanto_match_status=phone_context.tallanto_match_status,
)
```
Параметр `active_brand` **не передаётся**. Как следствие, `build_tallanto_live_card` получает `active_brand=None` (строка 261), и проверка бренд-мэтча (строки 261–265) не срабатывает. Клиент Фотона потенциально получит карточку с данными УНПК и наоборот — нарушение правила «бренды не смешиваются».  
Пример: контакт с `filial="mfti"` + `active_brand=None` → бренд-мисматч не отловится → `live_card.status="ok"`, `brand="unpk"` попадёт в промпт бота Фотона.

### D2. Обрезка `_payment_items`: поле `name` не включено
**Файл:** `tallanto_context.py` строки 327–340  
Платёж сжимается до `{date, status, sum}`. Поле `name` (назначение платежа / курс) отсутствует — бот не может сказать, за что именно был платёж.

### D3. Обрезка `_balance_items`: поле `name` (название абонемента) не включено
**Файл:** `tallanto_context.py` строки 343–359  
Баланс показывает `visits_left`, `status`, `valid_until` без названия абонемента. Клиент с несколькими абонементами не поймёт, к какому курсу относится остаток.

### D4. `_record_is_active` не обрабатывает числовой "0" как строку
**Файл:** `tallanto_context.py` строка 397  
```python
return status not in {"notactive", "inactive", "closed", "archive", "archived", "0"}
```
Строка `"0"` включена — правильно. Но если API вернёт целое `0` (не строку), то `_safe_text(0)` → `"0"` → обработается корректно. **Гипотеза** (не проверить без живого API): если Tallanto вернёт пустой статус для неактивной записи — такая запись попадёт в карточку (строка 396: `if not status: return True`).

### D5. Порча имени контакта при пустых частях ФИО
**Файл:** `tallanto_context.py` строки 33–42  
```python
"name": " ".join(...) or _safe_text(contact.get("name"))
```
Логика корректна — fallback на поле `name`. Но если `last_name="", first_name="", middle_name=""` и `name=""` — имя будет пустой строкой без алерта. В `live_card` поле `contact` не включается напрямую, но `_compact_contact` поставляет данные в `compact_contexts`. **Проверено:** порчи нет, просто тихое умолчание.

### D6. Лимит `_compact_items` = 20, лимит карточки отличается
**Файл:** `tallanto_context.py` строка 54  
`compact_contexts` режет `opportunities` и `requests` до 20 записей. В `live_card` `schedule` лимит 8, `payments` — 5, `balance` — 5. Несоответствие лимитов само по себе не баг, но нужно понимать: compact_contexts (для API-ответа и логирования) дают больше данных, чем live_card (для промпта).

### D7. `course_relations` загружаются, но в `live_card` не используются совсем
**Файл:** `tallanto_context.py` строки 143–146, 206–213  
`course_relations` собираются в `live_contexts` (строка 154–162), но в `build_tallanto_live_card` не используются — только `finances`, `abonements`, `class_relations`, `classes`. Данные `course_relations` попадают лишь в `compact_contexts`. Это бесполезный сетевой вызов для live_card.

---

## E. Проверено vs гипотеза

| # | Утверждение | Статус |
|---|---|---|
| E1 | `active_brand` не передаётся из `deals.py` в `build_deal_dossier` | **ПРОВЕРЕНО** — код строка 215–219 `deal_dossier.py`, аргумент отсутствует |
| E2 | `classes_by_ids` делает N отдельных HTTP-запросов в цикле | **ПРОВЕРЕНО** — `tallanto_api.py` строки 529–535 |
| E3 | `search_contacts_by_phone` делает до 40 запросов (5 полей × 8 кандидатов) | **ПРОВЕРЕНО** — строки 409–410 |
| E4 | `teacher_name` не попадает в `live_card` | **ПРОВЕРЕНО** — тест `test_tallanto_live_card.py` строка 38 + код `_schedule_and_enrollment` |
| E5 | `remaining_seats=10000` фильтруется как «безлимит» → пустая строка | **ПРОВЕРЕНО** — `_remaining_seats` строка 403 |
| E6 | SHD-филиал блокирует карточку, не причисляясь к Фотону | **ПРОВЕРЕНО** — `brand_scope_from_filial` строка 64 + тест |
| E7 | `opportunities` и `requests` загружаются, но в `live_card` не используются | **ПРОВЕРЕНО** — `live_contexts` строки 154–162 не включают эти поля |
| E8 | `course_relations` загружаются, но в `live_card` не используются | **ПРОВЕРЕНО** — то же |
| E9 | При пустом статусе `_record_is_active` возвращает `True` (активна) | **ПРОВЕРЕНО** — строка 396 |
| E10 | Потенциальное попадание неактивной записи при нечисловом нулевом статусе | **ГИПОТЕЗА** — требует живого теста с API |
| E11 | `TallantoApiClient` создаётся заново при каждом вызове (нет глобального кеша) | **ПРОВЕРЕНО** — строка 90 `tallanto_context.py` |
| E12 | `live_card` включается в промпт целиком без обрезки | **ПРОВЕРЕНО** — `deal_llm.py` строка 218 |
| E13 | Поле `name` абонемента/платежа не попадает в live_card | **ПРОВЕРЕНО** — код `_payment_items` и `_balance_items` |
