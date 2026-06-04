# План обработки `all_whatsapp_chats.txt` (v3 — встраивание в существующую инфраструктуру). 2026-05-29.

## Контекст и решения Дмитрия

- Есть второй MacBook со своим Claude + Кодексом, но **обработку ведёт Claude #2 на основном маке полностью** (вся работа на одном агенте).
- Хранилище — SQLite.
- **Шифровать PII не нужно**, без усложнений.
- В проекте уже есть свежие CRM-выгрузки и БД от Кодекса — Claude #2 должен в них разобраться **до** того, как изобретать структуры.

## Главное открытие: customer timeline уже существует

В `audits/for_claude_customer_timeline_dealaware_vs_control_20260516/` лежит **готовый пайплайн customer timeline** от Кодекса.

**Что уже выстроено и проверено:**
- источники: AMO (сделки), Tallanto (ученики/группы/звонки), `master_contacts_ru.csv`, `master_calls_ru.csv`;
- логика идентификации клиента: `normalized_phones` → `primary_phone` → `tallanto_ids`;
- формат импорта в CRM: `timeline_import_source.csv` с полями `matched_call_rows, matched_contact_rows, normalized_phones, primary_phone, risk_classes, selected_deal_*, tallanto_context_status, tallanto_ids`;
- готовый код: `code/context_provider.py`, `code/build_customer_timeline_*.py`, `code/*_sample_import.py`;
- тесты: 17 passed на контрольной выборке;
- результаты прошлого аудита: 100/100 timeline_found на контрольной, 18/100 ready_for_preview на сложной (узкое место — связка AMO↔Tallanto, не сам timeline).

**Tallanto-схема (из `tallanto_modules.json`):** Ученики, Организации, Заявки, Финансовые операции, Абонементы, Счета, Документы об образовании, История звонков IP телефонии, Сделки, Занятия, Предметы, Группы, Шаблоны абонементов, Записавшиеся на занятие, Записавшиеся в группу, Пользователи.

**Вывод:** WhatsApp **встраивается как ещё один источник** в существующий timeline-пайплайн. Не строим параллельную систему.

## Архитектура (с учётом существующего)

```
all_whatsapp_chats.txt
        │
        ▼ нормализатор (Этап 1)
        │
whatsapp_chats.sqlite (промежуточный слой: сами диалоги + индекс по phone)
        │
        ├──▶ Этап 2: матчинг с master_contacts_ru.csv (по primary_phone)
        │
        ├──▶ Этап 3: whatsapp_context_provider.py — по аналогии с готовым context_provider.py
        │           даёт по primary_phone набор matched_whatsapp_rows для timeline-импорта
        │           совместимо со схемой timeline_import_source.csv
        │
        └──▶ Этап 4: сезонная аналитика (отдельно от CRM-импорта, отчёты в markdown)
```

## Этапы

### Этап 0 — Разведка существующей инфраструктуры (Claude #2, 1 час)

Прежде чем что-то писать, Claude #2 читает и понимает:

1. `audits/for_claude_customer_timeline_dealaware_vs_control_20260516/README_FOR_CLAUDE.md` — что за пакет.
2. `audits/for_claude_customer_timeline_dealaware_vs_control_20260516/PROMPT_FOR_CLAUDE.md` — что от него хотели.
3. `audits/for_claude_customer_timeline_dealaware_vs_control_20260516/code/context_provider.py` — главный модуль.
4. `audits/for_claude_customer_timeline_dealaware_vs_control_20260516/code/build_customer_timeline_*_sample.py` — как строится timeline.
5. `audits/for_claude_customer_timeline_dealaware_vs_control_20260516/code/*_sample_import.py` — формат импорта в CRM.
6. `audits/for_claude_customer_timeline_dealaware_vs_control_20260516/dealaware_sample/deal_aware_timeline_import_source.csv` — пример формата.
7. `audits/for_claude_customer_timeline_dealaware_vs_control_20260516/dealaware_sample/source_manifest.json` — какие входные файлы используются.
8. `_local_archive_mango_api_downloads_20260507/product_appliance/tallanto_schema_check_20260509/tallanto_modules.json` + `tallanto_fields.json` — структура Tallanto.
9. `_local_archive_mango_api_downloads_20260507/product_appliance/tenants/foton/crm_snapshots/` — снимки CRM Фотона.
10. `TP UNPK DataExport_2026-05-21/` — выгрузка УНПК (заголовки таблиц).
11. `Contacts.xls` — контакты (заголовки).
12. Сами БД: `sqlite3 mango_mvp.db .schema`, `sqlite3 _local_archive_*/mango_product_appliance.sqlite .schema` — что внутри.
13. `audits/for_claude_customer_timeline_hard_control_20260516/README_FOR_CLAUDE.md` — hard control.

**Артефакт:** короткий конспект `whatsapp_step0_existing_infrastructure_2026-05-29.md` — что есть, к чему привязываемся, какие ограничения.

### Этап 1 — Нормализатор WhatsApp → SQLite (Claude #2, 1-2 часа)

Скрипт `scripts/whatsapp_normalize.py` (положить в проектные `scripts/`).

**Что делает:**
1. Парсит `all_whatsapp_chats.txt` (формат: блоки `===== CHAT: N =====`, дата/время отдельными строками, автор `You` = менеджер, индексы `0`/`1`/... = клиенты, голосовые → `Not supported WhatsApp internal message`).
2. Удаляет рекламные баннеры WhatsApp Business.
3. Склеивает многострочные реплики.
4. Извлекает дату/время в стандартный формат ISO.
5. **`brand_hint` по тексту менеджера** (только явные маркеры, без угадывания):
   - есть «УНПК», «МФТИ», «Менделеево» → unpk;
   - есть «Фотон», «Скорняжный», «Пацаева» → foton;
   - если в одном чате оба бренда — `mixed`, требует разметки;
   - иначе → null.
6. **PII без шифрования** (по решению Дмитрия): телефоны и имена хранятся в открытом виде в SQLite (БД остаётся локально, не уходит в git/Я.Диск).
7. Пишет в `whatsapp_chats.sqlite` (положить в `product_data/transcripts/`).

**Схема SQLite:**

```sql
CREATE TABLE chats (
  chat_id TEXT PRIMARY KEY,           -- N из блока ===== CHAT: N =====
  first_ts TEXT,
  last_ts TEXT,
  message_count INTEGER,
  brand_hint TEXT,                    -- foton | unpk | mixed | null
  brand_confidence REAL,
  client_phone TEXT,                  -- открытый, если найден
  client_name TEXT,                   -- открытый, если найден
  status TEXT                         -- raw | matched | analyzed
);

CREATE TABLE messages (
  msg_id INTEGER PRIMARY KEY AUTOINCREMENT,
  chat_id TEXT,
  ts TEXT,
  role TEXT,                          -- client | manager | service
  text TEXT,
  brand_hint TEXT,
  is_service_message INTEGER,
  FOREIGN KEY (chat_id) REFERENCES chats(chat_id)
);

CREATE TABLE crm_match (
  chat_id TEXT PRIMARY KEY,
  primary_phone TEXT,                 -- нормализованный +7XXXXXXXXXX
  tallanto_id TEXT,
  match_confidence TEXT,              -- high | medium | low
  match_basis TEXT,                   -- phone | phone+name | name_only
  matched_at TEXT
);

CREATE TABLE analyses (
  analysis_id INTEGER PRIMARY KEY AUTOINCREMENT,
  chat_id TEXT,
  analysis_type TEXT,
  result_json TEXT,
  created_at TEXT
);

CREATE INDEX idx_msg_chat ON messages(chat_id);
CREATE INDEX idx_msg_ts ON messages(ts);
CREATE INDEX idx_chat_phone ON chats(client_phone);
CREATE INDEX idx_match_phone ON crm_match(primary_phone);
```

**Выход:** `whatsapp_chats.sqlite` (~10-30 МБ), отчёт `whatsapp_normalize_report.json` (статистика).

### Этап 2 — Матчинг с CRM (Claude #2, 1-2 часа)

Скрипт `scripts/whatsapp_match_crm.py`.

**Источник истины для матчинга:** `master_contacts_ru.csv` + `master_calls_ru.csv` (используется в существующем timeline-пайплайне). Если файлы не найдены — Claude #2 разбирается, где они лежат сейчас (возможно `_external_handoffs/` или `_local_archive_*/`).

**Что делает:**
1. Нормализует номер из `chats.client_phone` → формат `+7XXXXXXXXXX`.
2. Ищет совпадение в `master_contacts_ru.csv`.
3. Если совпадение есть → проверяет ФИО ученика (если есть в тексте чата).
4. Заполняет `crm_match` с `confidence`:
   - `high` = phone+name совпали;
   - `medium` = только phone;
   - `low` = только частичное совпадение имени без phone.
5. CSV-отчёт `whatsapp_crm_match_review_<date>.csv` со всеми `medium`/`low` на ручную проверку Дмитрия.

### Этап 3 — Whatsapp context provider (Claude #2, 1-2 часа)

`scripts/whatsapp_context_provider.py` — **по аналогии с готовым `context_provider.py` из customer timeline пакета**.

**Что делает:**
1. На входе: `primary_phone`.
2. На выходе: набор `matched_whatsapp_rows` — список сообщений из WhatsApp за период, в формате, совместимом с `timeline_import_source.csv`.
3. Формат:

```python
{
  "primary_phone": "+7XXXXXXXXXX",
  "matched_whatsapp_rows": int,
  "whatsapp_first_ts": "...",
  "whatsapp_last_ts": "...",
  "whatsapp_brand_hint": "foton" | "unpk" | "mixed" | None,
  "whatsapp_summary": "Короткое резюме периода: темы, исход",  # генерация Haiku, не выдумка
  "risk_classes": [...]  # если попадают P0-маркеры
}
```

4. Этот провайдер можно дёргать из существующего `build_customer_timeline_*_sample.py` (по согласованию).

### Этап 4 — Сезонная аналитика (Claude #2, 2-3 сессии)

**Отдельно от CRM-импорта.** Это аналитические отчёты в markdown, не идут в продукт автоматически.

Артефакты в `D1_audit_backlog/`:

1. `whatsapp_seasonal_topics_2026-05-29.md` — топ-N вопросов по месяцу + бренду. Особый акцент:
   - **июнь-октябрь 2024** (нет звонков — единственный источник);
   - **июнь-октябрь 2025** (есть параллель со звонками).
2. `whatsapp_channel_comparison_2026-05-29.md` — WhatsApp vs звонки за июнь-октябрь 2025.
3. `whatsapp_real_p0_phrasings_2026-05-29.md` — реальные формулировки претензий/споров/жалоб; сверка с `p0_recall_spec.py`.
4. `whatsapp_manager_tone_2026-05-29.md` — обороты, эмодзи, длина реплик; материал для X2-калибровки.
5. `whatsapp_brand_leaks_managers_2026-05-29.md` — статистика бренд-утечек менеджеров; для тренинга команды.

### Этап 5 — Обогащение customer timeline (после Этапов 1-4, отдельная итерация)

После того как Claude #2 закончит — Claude #1 (в новом диалоге) пишет ТЗ Кодексу main на:
- интеграцию `whatsapp_context_provider.py` в основной timeline-пайплайн;
- расширение схемы `timeline_import_source.csv` полями `matched_whatsapp_rows, whatsapp_first_ts, whatsapp_last_ts, whatsapp_brand_hint, whatsapp_summary`;
- регресс-тесты по аналогии с уже существующими 17 тестами customer timeline.

**Заливка в CRM** — НЕ автоматическая, по правилу проекта. Дмитрий грузит батчем через интерфейс или подтверждает CSV.

## Что НЕ делать

- НЕ изобретать параллельную инфраструктуру timeline — использовать существующий пайплайн.
- НЕ грузить сырой 17 МБ файл в один диалог Claude.
- НЕ автоматически писать в AMO/Tallanto.
- НЕ угадывать бренд (только по явным маркерам).
- НЕ выкатывать аналитику без твоей сверки.
- НЕ блокировать M1-прогон и пилот.

## Где хранятся артефакты

| Артефакт | Путь |
|---|---|
| Скрипты | `scripts/whatsapp_*.py` |
| БД | `product_data/transcripts/whatsapp_chats.sqlite` |
| Отчёты Claude #2 | `D1_audit_backlog/whatsapp_*.md` |
| CRM-match review | `audits/_inbox/whatsapp_crm_match_<date>/` |
| Тесты | `tests/test_whatsapp_*.py` |

## Стартовый промт для Claude #2

> Привет. Тебя зовут как Claude #2 — аналитик. У нас новая задача: обработать выгрузку `~/Projects/Mango analyse/all_whatsapp_chats.txt` (17 МБ, 366К строк) и встроить её в существующий customer timeline пайплайн.
>
> **Прочитай сначала, в этом порядке:**
>
> 1. `~/Projects/Mango analyse/CLAUDE.md` — правила проекта.
> 2. `~/Projects/Mango analyse/D1_audit_backlog/whatsapp_processing_plan_2026-05-29.md` — твой план работы со всеми деталями.
> 3. `~/Projects/Mango analyse/audits/for_claude_customer_timeline_dealaware_vs_control_20260516/README_FOR_CLAUDE.md` — пакет существующего customer timeline.
> 4. `~/Projects/Mango analyse/audits/for_claude_customer_timeline_dealaware_vs_control_20260516/code/context_provider.py` — главный модуль, к которому ты по аналогии будешь делать `whatsapp_context_provider.py`.
>
> **Главное правило:** не изобретай параллельную инфраструктуру. Customer timeline уже существует. Твоя задача — встроить WhatsApp как ещё один источник.
>
> Делаешь 5 этапов из плана: разведка → нормализатор → матчинг с CRM → context provider → сезонная аналитика. После каждого этапа — короткий отчёт мне (Дмитрию) с тем, что нашёл и что сделал.
>
> **Стиль:** только русский, кратко и по делу, без английских терминов, проверяй каждый вывод до выдачи. PII можно хранить открыто (БД остаётся локально). Бренды Фотон и УНПК НИКОГДА не смешиваются.
>
> Время на полный объём: ~3-5 сессий в твоём диалоге (один диалог = до переполнения контекста, потом стартуешь свежий со своим отчётом-передачей).
>
> Поехали.
