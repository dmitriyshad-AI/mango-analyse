# ТЗ-12 (для диалога D4 «KB: Исторические каналы»): история клиента + профиль v1 — большой блок под флагами

Версия 2 (после аудита второго архитектора: FAIL→правки внесены→готово к выдаче). Утверждено Дмитрием 10.06: реализацию ведём СЕЙЧАС, до старта пилота, но так, чтобы на пилот не влиять: новые модули отдельными файлами, касания общих модулей минимальны и за NEG «поведение без новых вызовов байт-в-байт». Архитектор трека — Claude (existing_clients); опора: D1_audit_backlog/existing_clients/INVENTORY_timeline_dealaware_2026-06-10.md и STATE_and_PLAN_existing_clients_2026-06-10.md.

**Две поставки** (рекомендация аудита, принята): ПОСТАВКА 1 = РП-0+РП-1+РП-2 (гигиена, мета, замер — короткая, разблокирует решение по модели); ПОСТАВКА 2 = РП-3…РП-7 (инфраструктура профилей). Отчёт в _done по каждой поставке отдельно.

**Политика брендов (решение Дмитрия 10.06):** хранение истории/профиля ОБЩЕЕ по обоим брендам (один tenant_id="foton" — норма), бот «знает всё»; разделение брендов — фильтр на ВЫХОДЕ бота (уже существует, не трогаем). Бренд хранится как атрибут происхождения события/поля.

## Жёсткие границы (нарушение = стоп)

1. НЕ трогать: `src/mango_mvp/channels/*`, `src/mango_mvp/integrations/draft_loop.py` и amo_wappi_*, `pilot_context_assembly.py`, профиль pilot_gold_v1, снимок базы знаний. Никаких импортов из новых модулей в channels/.
2. Никаких live-вызовов AMO/Tallanto/Wappi и отправок. Запись только в НОВУЮ локальную базу профилей и в timeline-базу через существующий механизм импорта.
3. ПДн: новые базы/выгрузки — вне git. В составе РП-0 добавить в .gitignore: `all_whatsapp_chats.txt` (сейчас НЕ игнорится — проверено git check-ignore, rc=1) и `product_data/customer_profiles/`. Пункт приёмки каждого РП: `git status` чистый после прогона.
4. ASR не запускать. LLM-вызовы в этом ТЗ есть ТОЛЬКО в РП-2 (A/B analyze, разрешение Дмитрия дано 10.06) — на готовых расшифровках.
5. Полный pytest перед отчётом (не подмножество). Каждый РП — отдельный коммит со своими тестами и NEG.

## Состав блока (7 рабочих пакетов, порядок = порядок коммитов)

### РП-0. Гигиена git (мелкий, первый)
.gitignore: + `all_whatsapp_chats.txt`, + `product_data/customer_profiles/`. Семантика проверки (чтобы не перепутать): СЕЙЧАС `git check-ignore all_whatsapp_chats.txt` возвращает rc=1 (НЕ игнорится — риск утечки ПДн одним `git add -A`); ПОСЛЕ правки NEG-тест: rc=0 (игнорится) для обоих путей.

### РП-1. Фиксация модели/промпта в analysis_json
Сейчас модель НЕ записывается (пробел инвентаризации §1.2) — без этого перепрогон создаст неразличимую смесь. Точка: `src/mango_mvp/services/analyze.py:2307-2308` (там `analysis = self._normalize_analysis(...)` и присвоение `call.analysis_json`). После нормализации добавить блок мета:

```python
analysis["analysis_meta"] = {
    "analysis_model": <модель из конфига, которой реально шёл вызов: codex→settings.codex_analyze_model / openai→settings.openai_analysis_model>,
    "analysis_provider": <"codex_cli" | "openai" | "mock">,
    "analysis_prompt_version": <ANALYZE_PROMPT_VERSION_COMPACT("v6") | _FULL("v7") — фактически использованная (analyze.py:160-161)>,
    "analyzed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
}
```

`migrate_analysis_payload` (вызывается из cli.py:1116 и export_ai_office.py:84) НЕ ТРОГАТЬ: он переписывает ИСТОРИЧЕСКИЕ payload — мета там стала бы ложью (звонок считала старая модель). Мета добавляется ТОЛЬКО в свежий analyze-проход (точка 2307). Требования: СТАРЫЕ analysis_json не мигрировать; `analysis_schema_version` не менять. Тесты: юнит «после analyze у записи есть analysis_meta с непустой моделью»; NEG «migrate_analysis_payload НЕ добавляет analysis_meta» (оба вызывающих пути); NEG «mock-провайдер пишет analysis_provider=mock».

### РП-2. Сравнительный замер моделей на выжимках (A/B, прогон включён в ТЗ)
Скрипт уже есть: `scripts/run_analyze_ab_test.py` (--source-db, --ids-file, --models nargs=+, --prompt-profile compact|full уже параметризованы — проверено). Доработки (обязательны, аудит): (а) `summarize_call()` (строка 154) НЕ читает analysis_meta — добавить в выход каждой строки поля analysis_model и analysis_prompt_version из `parsed["analysis_meta"]` (после РП-1); (б) отчёт-матрица покрытия со столбцом prompt_version.

Выборка — 50 содержательных звонков из canonical-базы (read-only), полуфабрикат SQL:

```sql
SELECT call_id FROM canonical_calls
WHERE analysis_status='done' AND duration_sec>=120
  AND analysis_json NOT LIKE '%non_conversation%'
  AND started_at>='2026-01-01'
ORDER BY random() LIMIT 50;
```

Плечи (каждое — тот же набор 50): 1) gpt-5.4-mini + промпт compact v6 = базлайн (текущее состояние); 2) gpt-5.4-mini + промпт full v7 (проверка «дело в промпте?»); 3) gpt-5.4 + v6; 4) gpt-5.5 + v6 — если модель доступна в CLI, иначе пропустить с пометкой в отчёте. Кэш не мешает (ключ включает модель — llm_response_cache.py:24-34). Расшифровки/resolve НЕ трогаются.

Отчёт `audits/_inbox/ab_analyze_<дата>/`: матрица по плечам — % заполненности target_product, next_step.action, objections, history_summary (длина/наличие); таблица расхождений по 10 случайным звонкам (старое поле/новое поле); БЕЗ сырых телефонов/ФИО в отчёте (маскирование как в read_api). Решение «какой моделью перепрогонять зону» НЕ принимать — это регрейд Claude + решение Дмитрия. Критерии успеха замера (зафиксированы ДО прогона, план §2): полнота product/next_step на содержательных растёт, 0 выдумок в именах/числах (проверит регрейд), выжимка не хуже. budget — только справочно, НЕ критерий.

### РП-3. Хранилище профилей + детерминированный сборщик из timeline
Новый модуль `src/mango_mvp/customer_profile/` (store.py, contracts.py, builder.py, build_cli.py). Никакого LLM: профиль собирается ДЕТЕРМИНИРОВАННО из уже извлечённых полей analysis_json/Tallanto/AMO-событий timeline-базы. База: `product_data/customer_profiles/customer_profiles.sqlite` (в .gitignore по РП-0), read-only вход — `product_data/customer_timeline/canonical_readonly_20260521_v5/customer_timeline.sqlite` (и будущие пересборки v6+).

DDL (полуфабрикат):

```sql
CREATE TABLE customer_profiles (
  profile_id TEXT PRIMARY KEY,        -- = customer_id из timeline
  tenant_id TEXT NOT NULL,
  primary_phone TEXT, display_name TEXT,
  built_at TEXT NOT NULL, build_id TEXT NOT NULL,
  source_event_count INTEGER NOT NULL,
  last_event_at TEXT
);
CREATE TABLE profile_fields (
  field_id TEXT PRIMARY KEY,
  profile_id TEXT NOT NULL REFERENCES customer_profiles(profile_id),
  field TEXT NOT NULL,                -- parent_name|child_name|grade|subject|format|target_product|next_step|objection|brand_touch|tallanto_balance|tallanto_group|payment_fact|...
  value TEXT NOT NULL,
  child_key TEXT NOT NULL DEFAULT '', -- дети раздельно (механика ТЗ-8)
  brand TEXT NOT NULL DEFAULT 'unknown',  -- бренд-источник значения (foton|unpk|unknown)
  source_system TEXT NOT NULL, source_ref TEXT NOT NULL,
  event_at TEXT NOT NULL,             -- когда факт прозвучал (НЕ когда собрали)
  quote TEXT NOT NULL DEFAULT '',     -- цитата-носитель ≤200 симв., если есть
  superseded_by TEXT NOT NULL DEFAULT ''  -- field_id более позднего значения; история не удаляется
);
CREATE INDEX idx_profile_fields_lookup ON profile_fields(profile_id, field, child_key, superseded_by);
CREATE TABLE profile_builds (build_id TEXT PRIMARY KEY, started_at TEXT, finished_at TEXT, timeline_db_path TEXT, timeline_db_sha256 TEXT, profiles_built INTEGER, notes TEXT);
```

Правила builder (контракт): профиль ПЕРЕСОБИРАЕТСЯ целиком из событий (идемпотентно; никаких правок на месте); «позднее побеждает» = у конфликтующих значений одного (field, child_key) старое получает superseded_by, не удаляется; при РАВНЫХ event_at побеждает значение с большей confidence источника, при равенстве — лексикографически больший source_ref (детерминизм важнее «правильности» — конфликт всё равно виден в истории); дети раздельно по child_key (child_1/child_2... в порядке появления; имя ребёнка — отдельное поле того же child_key); каждое поле обязано иметь source_system+source_ref+event_at — поле без происхождения не пишется (жёсткий инвариант, тест).

**ДВА входа builder (критическая правка аудита, подтверждена по сырью):** в timeline-событиях звонков структурных полей НЕТ — record_json несёт только {brand, contentful, duration_sec, manual_review_required} (проверено запросом по v5, 200 строк, structured_fields=0). Поэтому: ВХОД-1 = timeline sqlite (идентичности, identity_links, Tallanto/AMO-события, бренд звонка — в record.brand: foton 8 714 / unpk 10 906 / unknown 45 266); ВХОД-2 = `stable_runtime/canonical_master_20260523_audio_working_store_v1/canonical_calls_master.db` read-only (СТРУКТУРНЫЕ поля из analysis_json: structured_fields/people/student/interests, target_product, next_step, objections — ключи как в analyze.py:1960-1982). Связка входов: нормализованный телефон (identity_links link_type=phone ↔ canonical_calls.phone; нормализация utils/phone, допускается матч по хвосту 10 цифр) — у master-базы НЕТ call_id, телефон + started_at и есть ключ; event_at поля = started_at звонка. Звонок без телефона или телефон без identity → счётчик unmatched в отчёте, поле не пишется. Quote для полей из analysis_json v1 = пустая строка (цитат там нет — честно не выдумываем; quote живёт у полей из будущих диалоговых источников). Источники v1: ВХОД-2 (звонки), `tallanto_snapshot` (баланс/группа/посещения), `amocrm_snapshot` (статусы сделок). Бренд поля: для звонков = record.brand события; для прочих = relevance_tags, иначе unknown.

CLI: `build_cli.py --timeline-db ... --profiles-db ... --all | --customer-id ... | --phone ...` и `--show-phone` (печать профиля в консоль с маскированием). Тесты: builder на синтетической timeline-базе (фикстура), конфликт «класс 7→8» даёт superseded, два ребёнка не смешиваются, поле без происхождения падает, идемпотентность (два прогона = одинаковый результат), build_id/sha256 заполнены.

### РП-4. Импорт Telegram-истории в timeline (профильная зона D4)
Источник: `telegram_exports (2)/<выгрузка>/` — формат подтверждён: `dialogs.jsonl` (dialog_id, peer_kind, is_user, username, phone) + `messages.jsonl` (dialog_id, dialog_name, message_id, date ISO, sender_id, text, out, sender_username, sender_phone). Путь в timeline уже существует: `ChannelMessageNormalizer` (ingestion.py:555) ждёт payload {channel, channel_thread_id, channel_message_id, channel_user_id, received_at, direction, display_name, text}.

Новый скрипт `scripts/import_telegram_export_to_timeline.py` поверх существующего import_cli (source_kind=channel_snapshot). Полуфабрикат маппера:

```python
def tg_message_to_payload(msg: dict, dialog: dict, brand: str) -> dict:
    return {
        "channel": "telegram",
        "channel_thread_id": str(msg["dialog_id"]),
        "channel_message_id": str(msg["message_id"]),
        "channel_user_id": str(msg.get("sender_id") or msg["dialog_id"]),
        "received_at": msg["date"],
        "direction": "outbound" if msg.get("out") else "inbound",
        "display_name": msg.get("dialog_name") or dialog.get("title") or "",
        "text": (msg.get("text") or "").strip(),
        "brand_hint": brand,  # из манифеста выгрузки, не из текста
    }
```

Требования: только peer_kind=user (группы/каналы скип со счётчиком); пустые text скип; сшивка с существующим клиентом — если в dialogs.jsonl есть phone (формат Telethon: строка цифр БЕЗ «+», часто null — у скрытых настройками приватности) → нормализовать через utils/phone и добавить IdentityLink phone (точный матч), иначе остаётся channel_session_id (INFERRED — как в ingestion.py:594-618); бренд выгрузки — параметр CLI `--brand foton|unpk|unknown` (какому аккаунту принадлежит экспорт, подтверждает Дмитрий), кладётся в relevance_tags. Идемпотентность через dedupe_key (контракт contracts.py:722-732: {tenant}:{source_system}:{event_type}:{source_id} — source_id у нас «telegram:<message_id>» уникален в рамках диалога, поэтому в source_id включить и thread: `telegram:<dialog_id>:<message_id>`). Выход: отчёт-счётчики (диалогов/сообщений/импортировано/скипнуто/сшито по телефону/только session). Тесты: маппер на СИНТЕТИЧЕСКИХ фикстурных jsonl по форматам Приложения А (реальные выгрузки в тесты не класть — ПДн); повторный импорт не дублирует (счётчик дублей dedupe); группа скипается; направление out→outbound; phone=null не падает.

### РП-5. Импорт WhatsApp-истории в timeline
Источник: `all_whatsapp_chats.txt` (17 МБ). Формат (проверен): блоки `\n===== CHAT: <имя> =====\n`, затем строки даты `2025-04-30`, времени `09:20`, отправителя (`You` = исходящее, иначе имя/номер контакта), текст до следующего тайм-штампа; служебное `Not supported WhatsApp internal message` — скип. Референс разбора строк — scripts/whatsapp_extract_real_questions.py (формат тот же). Парсер-каркас:

```python
CHAT_HDR = re.compile(r"^===== CHAT: (.+?) =====$")
DATE_RE  = re.compile(r"^\d{4}-\d{2}-\d{2}$")
TIME_RE  = re.compile(r"^\d{2}:\d{2}$")
SKIP_RE  = re.compile(r"Not supported WhatsApp internal message")
# конечный автомат: header → (date)* → time → sender → text* ; messageid = f"{chat}:{date}T{time}:{n_в_минуте}"
```

channel="whatsapp", thread = имя чата (если имя — телефон, нормализовать и дать IdentityLink phone), direction: You→outbound. Многострочный текст: строки между «отправителем» и следующим TIME_RE/DATE_RE/CHAT_HDR склеиваются через \n в одно сообщение. Бренд: `--brand` параметром (один файл = один аккаунт Wappi). Те же счётчики/идемпотентность/тесты, что РП-4. Фикстура для тестов — СИНТЕТИЧЕСКИЙ файл по формату Приложения Б (10-15 сообщений: заголовок чата, многострочное, You, служебное, два чата) с известным эталонным счётчиком; реальный файл в тесты не класть (ПДн). Особый NEG: парсер не теряет сообщения (счётчик сообщений фикстуры = эталон) и на полном файле выдаёт счётчики в отчёт.

### РП-6. Событийная пересборка профилей (зачаток событийной модели)
Новый `scripts/refresh_customer_profiles.py`: режим `--since <ts>` — найти клиентов с новыми timeline-событиями после ts (по created_at событий) и пересобрать ТОЛЬКО их профили; режим `--from-journal <path копии journal.jsonl>` — детект «диалог затих»: полуфабрикат:

```python
def detect_quiet_dialogs(journal_rows, now_utc, quiet_minutes=30):
    last = {}
    for r in journal_rows:  # rows() уже парсит jsonl — draft_loop.py:127-151
        key = (r.get("profile_id"), r.get("chat_id"))
        ts = parse_iso(r.get("created_at"))
        if key[0] and key[1] and ts: last[key] = max(last.get(key, ts), ts)
    return [k for k, ts in last.items() if (now_utc - ts) >= timedelta(minutes=quiet_minutes)]
```

ВАЖНО: журнал читать из КОПИИ (путь параметром; боевой `~/.mango_local/draft_loop/journal.jsonl` не трогать и не лочить); сам draft_loop НЕ модифицируется и его резолвер НЕ вызывается (он ходит в живой AMO — здесь запрещено). Сшивка затихшей пары с клиентом — ДЕТЕРМИНИРОВАННО через identity_links timeline: link_type='telegram_user_id' c link_value=chat_id ИЛИ link_type='channel_session_id' c link_value='telegram:<chat_id>' (формат — ingestion.py:599,611; после РП-4 такие связи появятся из импорта). Нет совпадения → пара в отчёт unmatched (с давностью затишья), профиль не трогаем. Выход: список затихших пар + пересборка профилей сшитых. Тесты на фикстурном журнале (затихший/активный/битые строки/несшитый chat_id).

### РП-7. Выжимка профиля для карточки CRM — только генерация (записи в CRM в этом ТЗ НЕТ)
`customer_profile/crm_summary.py`: детерминированный текст ≤1200 симв. из активных (не superseded) полей: шапка (имя, телефон маскированный, дети: имя-класс-предметы), статусы (Tallanto группа/баланс, последняя оплата), последние договорённости (next_step с датой источника), история касаний по брендам (счётчики: звонков/сообщений по каналам, период). Полная история обоих брендов — это ВНУТРЕННЯЯ карточка для менеджера (решение Дмитрия 10.06), бренд-фильтр не применяется; но внутрь НЕЛЬЗЯ: юр.реквизиты, source_id/служебные ключи, цитаты длиннее 200. CLI `--phone → текст в консоль/файл предпросмотра`. Тесты: лимит длины, маскирование телефона, superseded не попадает, пустой профиль → внятный текст-заглушка.

## Приёмка блока

1. Полный pytest зелёный + все NEG из РП. 2. `git status` чистый после всех прогонов (ПДн не утекли). 3. Импорты: счётчики TG (≥1 выгрузка) и WA, доля сшитых по телефону. 4. Профили построены для всей v5-базы (~16к) без падений; отчёт: время сборки, профилей, полей, конфликтов-superseded; 5 обезличенных примеров профиля + 3 примера CRM-выжимки в отчёте. 5. A/B-отчёт с матрицей в audits/_inbox (без решения — решение за Дмитрием после регрейда Claude). 6. Отчёт блока в tasks/_done с перечнем коммитов. 7. Pilot-инвариант: ни одного нового импорта в channels/ (grep в NEG-тесте), поведение бота байт-в-байт (существующие смоук-тесты канала зелёные без изменений).

## Очерёдность и вопросы

ПОСТАВКА 1: РП-0→1→2 (РП-2 строго после РП-1, чтобы мета писалась). ПОСТАВКА 2: РП-3→4→5→6→7 последовательно. Вопросы Дмитрию (через главный диалог, не блокируют РП-0..3): бренд каждой TG-выгрузки и WA-файла (для --brand); quiet_minutes по умолчанию (предлагаю 30).

## Приложение А. Формат Telegram-экспорта (структура подтверждена по реальной выгрузке, значения обезличены)

dialogs.jsonl, одна строка = диалог:
```json
{"dialog_id": 123456789, "name": "Имя Фамилия", "peer_kind": "user", "unread_count": 0, "folder_id": null, "is_user": true, "is_group": false, "is_channel": false, "top_message_id": 16646, "top_message_date": "2026-04-01T14:32:42+00:00", "first_name": "Имя", "last_name": "Фамилия", "username": null, "phone": null}
```
messages.jsonl, одна строка = сообщение:
```json
{"dialog_id": 123456789, "dialog_name": "Имя Фамилия", "peer_kind": "user", "message_id": 16520, "date": "2026-03-28T09:38:18+00:00", "sender_id": 123456789, "text": "Добрый день! Подскажите курсы по математике для 6 класса", "out": false, "reply_to_msg_id": null, "has_media": false, "media_path": null, "media_error": null, "sender_username": null, "sender_phone": null}
```
Выгрузка `telegram_exports (2)/local_vm_2024-04-01_with_contacts/`: 600 диалогов, 7 708 сообщений; media/ игнорировать в v1 (только text). phone у большинства null (приватность) — это норма, сшивка тогда по session.

## Приложение Б. Формат all_whatsapp_chats.txt (структура подтверждена, значения обезличены)

```
===== CHAT: <имя или номер> =====

Whatsapp - <имя>
Chat history with <имя>
2025-04-30          ← строка-дата (контекст для последующих сообщений)
09:20               ← время сообщения
You                 ← отправитель: You=исходящее, иначе имя/номер клиента
Текст сообщения, может быть
в несколько строк подряд
2025-06-05          ← новая дата
14:35
<имя клиента>
Ответ клиента
```
Служебные строки «Not supported WhatsApp internal message» — скип со счётчиком. Референс построчного разбора — scripts/whatsapp_extract_real_questions.py (тот же формат). Файл 17 МБ, локальный, после РП-0 в .gitignore.
