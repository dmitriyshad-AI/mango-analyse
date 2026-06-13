# Track 1 — Профили клиентов: дубль-дети, склейка, бренды каналов

**Дата аудита:** 2026-06-13  
**Аудитор:** субагент (read-only, без записи)  
**Охват:** tz12_batch3 (старая v6), tz16_v7_20260612 (новая v7), tz21_after_tail_20260613  

---

## A. Карта модулей профилей

### Назначение и точки входа

| Модуль | Путь | Назначение |
|---|---|---|
| `CustomerProfileBuilder` | `src/mango_mvp/customer_profile/builder.py` | Ядро сборки: читает timeline → строит profile_fields |
| `CustomerProfileSQLiteStore` | `src/mango_mvp/customer_profile/store.py` | Запись/чтение профилей в SQLite (schema v1) |
| `ProfileFieldCandidate` / `ProfileSnapshot` | `src/mango_mvp/customer_profile/contracts.py` | Контракты данных; superseded-правила; normalize_brand |
| `crm_summary.py` | `src/mango_mvp/customer_profile/crm_summary.py` | Рендер выжимки профиля для бота (≤1200 символов) |
| `build_cli.py` | `src/mango_mvp/customer_profile/build_cli.py` | CLI: `--all / --phone / --customer-id` |
| `build_tz16_profiles_v7.py` | `scripts/build_tz16_profiles_v7.py` | Пакетный прогон: source=tz12_batch3 → out=tz16_v7; микро(5) + полный + idempotence |
| `refresh_customer_profiles.py` | `scripts/refresh_customer_profiles.py` | Инкрементный refresh: `--since ISO` или `--from-journal` (quiet=30 мин) |
| `customer_context_for_draft.py` | `src/mango_mvp/channels/customer_context_for_draft.py` | Контекст для черновика; два режима: legacy-mapping и prompt-ready dict |

### Запуск

- **Полная пересборка:** `scripts/build_tz16_profiles_v7.py --out-root <dir>` (99 сек на 18399 профилей, идемпотентен)
- **Инкрементный:** `refresh_customer_profiles.py --timeline-db X --profiles-db Y --since ISO` или `--from-journal journal_copy.jsonl`
- **По звонку/телефону:** `build_cli.py --phone +7...` (для live-пилота)

---

## B. Поток данных профиля

```
canonical_calls_master.db (звонки v7)
  ↓ _fields_from_master_calls
  └─ телефон→profile_id через identity_links (timeline)
  └─ brand → brand_index (из mango_call events в timeline)
  └─ call_analysis_fields → child_fields + parent_name/format/target_product/next_step/objection

customer_timeline.sqlite
  ↓ _fields_from_timeline
  └─ event_types: tallanto_student_snapshot, amo_deal_stage, amo_contact_snapshot
  └─ brand_from_payload: payload.metadata.brand → payload.record.brand → payload.brand_hint

apply_child_slot_merge_candidates (в памяти)
  └─ collect_child_slots → child_slot_groups (union-find по normalize_name_tokens)
  └─ rekey дублей на canonical child_key
  └─ emit child_slot_merge_candidate markers

apply_superseded_rules (в памяти)
  └─ winner = last event_at + SOURCE_CONFIDENCE + source_ref

CustomerProfileSQLiteStore.replace_profiles (DELETE + INSERT, атомарно по profile_id)
  → customer_profiles.sqlite
```

**Три таблицы в профильной БД:** `customer_profiles`, `profile_fields`, `profile_builds`.  
Индекс: `idx_profile_fields_lookup ON profile_fields(profile_id, field, child_key, superseded_by)`.

---

## C. Находки неэффективности

### C1. Общий child_key для разных профилей — «phantom collision» через stable_child_key
**Файл:** `builder.py`, строки 432–436 (`stable_child_key`)  
**Суть:** `stable_child_key` берёт SHA256[:8] от нормализованного имени/класса/предмета ребёнка. Когда запись содержит ТОЛЬКО класс (без имени), ключ = SHA256("9")[:8] = `child_19581e27`. Этот один child_key встречается в **747 профилях** (1728 строк). SHA256("математика; физика")[:8] = `child_1c05a18f` — 683 профиля.  
**Последствие:** значение child_key само по себе не несёт уникальности между профилями — оно уникально только внутри одного profile_id. Коллизий межпрофильных нет (поле `profile_id` всегда в запросах). Неэффективность: при дебаггинге — мусор при grep по child_key без profile_id.

### C2. Superseded rate = 38% (72 791 из 190 614 строк)
**Файл:** `contracts.py`, строки 120–140 (`apply_superseded_rules`)  
**Суть:** Победитель — последнее событие + SOURCE_CONFIDENCE. Проигравшие хранятся со ссылкой на победителя. 38% строк — мёртвый груз. При полной пересборке (replace_profiles делает DELETE+INSERT) это норма, но растёт при каждом добавлении новых звонков без очистки старых.

### C3. crm_summary.py: поиск по телефону — полный скан таблицы
**Файл:** `crm_summary.py`, строки 169–186 (`_profile_ids_by_phone`)  
**Суть:** Нет индекса на `primary_phone`. Делается `SELECT ALL` customer_profiles и сравнивает нормализованный телефон в Python. При 18 399 профилях — приемлемо, но не масштабируется.

### C4. child_name с ФИО — затрудняет merge
**Файл:** `builder.py`, строки 398–414 (`child_fields`)  
**Суть:** `child_name` принимает `child.get("child_name") or child.get("name")` из analysis JSON. LLM-анализатор часто пишет туда полное ФИО (Фамилия Имя Отчество). Обнаружено 3671 записей с 2+ словами в child_name (из 16 401 active). `normalized_name_tokens` пытается разобрать такие строки, но `_DIMINUTIVE_CANONICAL` покрывает только отдельные имена. При ФИО из 3 слов `known_tokens` пытается найти каждый токен в словаре — если ни один не найден (напр. нестандартное имя), `child_name_keys` вернёт пустое множество → `child_slot_name_key` вернёт `""` → `child_slots_match` вернёт False → **слоты не сольются**, даже если это один и тот же ребёнок.

---

## D. Находки порчи / обрезки / дублей

### D1. КРИТИЧНО: 63% профильных полей имеют brand=unknown (81 332 из 117 823 активных)
**Источник проверен:** `profile_fields` + `timeline_events`.  
**Причина:** Все поля из звонков (`source_system=mango_processed_summary`) получают бренд через `brand_index`, который строится по `mango_call` events в timeline. Из 64 886 mango_call событий — 45 266 (70%) имеют `metadata.brand=unknown` и `record.brand=unknown`. Из них 10 906 — brand=unpk, 8 714 — brand=foton. Итого: ~30% звонков брендированы, 70% — нет. Поля из CRM-снапшотов (tallanto + amo) тоже преимущественно unknown: 13 597 из 16 239 tallanto_student_snapshot = unknown. Результат: бот получает контекст профиля без понимания, к какому бренду относится большинство данных.

### D2. «Phantom слоты» у активных клиентов — до 19 child_key на один профиль
**Источник проверен:** DB_NEW, таблица profile_fields.  
**Пример (обезличенный):** профиль с 409 событиями имеет 19 child_keys при active_fields=0 (супрессия всех). Другой с 349 событиями — 7 child_slots, все brand=foton. Профиль с 18 слотами: среди них `child_19581e27` (класс=9), `child_1c05a18f` (физика+математика без имени), и несколько именованных.  
**Причина:** Каждый звонок порождает новый child_key из stable_child_key, если имя не нормализуется в одно и то же каноническое. Слоты без имени (`child_{sha256_of_grade}`) не сливаются через `child_slots_match` — она требует совпадение `child_name_keys`, а у безымянных слотов это пусто.

### D3. Двойные записи grade в одном child_key
**Источник проверен:** профиль с 18 слотами выше.  
**Пример:** `child_181165cf` имеет value для grade = "дошкольник, 6 класс, 3 класс" в одном поле. Это происходит, когда `timeline_field_values` собирает `group_name` и `group` из Tallanto-снапшота, а значение уже содержит несколько классов конкатенацией.

### D4. 1 053 звонка без совпадения профиля (unmatched_calls)
**Источник:** `summary.json`, строка `unmatched_calls: 1053`.  
**Суть:** Звонки из canonical_calls_master, телефон которых не нашёлся в identity_links. Поля этих звонков не попадают ни в один профиль — тихая потеря данных без алерта.

### D5. 77 звонков из blacklist не имеют v7-анализа (`blacklist_ids_with_v7: 0`)
**Источник:** `summary.json`, analysis_counts.  
**Суть:** 77 ID из blacklist_77.txt присутствуют в master_calls_db, но все без v7-разбора. Если эти звонки соответствуют активным профилям, их профильные поля строятся на старом non-v7 анализе (или отсутствуют совсем). Это потенциальное расхождение качества данных.

### D6. Profile с 409 событиями и 0 активных полей
**Источник:** `summary.json` (profile_example_1: source_event_count=409, active_field_count=0), подтверждено DB.  
**Суть:** Самый «богатый» по событиям профиль не имеет ни одного активного поля. Возможные причины: все звонки non-contentful (неразборчиво/автоответчик), нет Tallanto/AMO снапшота, или все поля superseded. Требует ручной проверки.

### D7. Merge срабатывает только на имя, игнорирует класс и предмет
**Файл:** `builder.py`, строки 593–596 (`child_slots_match`)  
**Суть:** `child_slots_match` возвращает True ТОЛЬКО если `child_slot_name_key` совпадает по обеим сторонам — т.е. требует непустого нормализованного имени у обеих записей. Если у одного слота есть имя, у другого только класс/предмет — слоты НЕ сольются. Это намеренный выбор безопасности (не сливать по неполным данным), но означает, что большинство «phantom слотов» (без имени) остаются несёрными и загромождают профиль.

---

## E. Проверено vs гипотеза

### ПРОВЕРЕНО (опорой на файл+строку или данные)

| # | Утверждение | Как проверено |
|---|---|---|
| 1 | Схема БД идентична в OLD и NEW — те же 3 таблицы, тот же DDL | Read store.py + python3 query обеих БД |
| 2 | 18 399 профилей в OLD = NEW = TZ21 | python3 COUNT(*) |
| 3 | child_19581e27 = SHA256("9")[:8], child_1c05a18f = SHA256("математика; физика")[:8] | python3 hashlib |
| 4 | child_19581e27 встречается в 747 профилях | COUNT(DISTINCT profile_id) |
| 5 | 63% активных полей = brand:unknown (81332/117823) | GROUP BY brand |
| 6 | Источник brand:unknown — mango_call events: 45 266 из 64 886 имеют metadata.brand=unknown | GROUP BY в timeline |
| 7 | CRM-снапшоты (tallanto/amo) тоже преимущественно unknown: 13597/16239 tallanto unknown | GROUP BY в timeline |
| 8 | Superseded rate = 38% (72791 строк) | summary.json + COUNT |
| 9 | 1053 звонка unmatched (тихая потеря) | summary.json |
| 10 | Merge работает ТОЛЬКО по child_name, не по grade/subject | builder.py строки 593-596 |
| 11 | child_name содержит 3671 записи с 2+ словами (в т.ч. ФИО) | GROUP BY len(split) |
| 12 | Profile с 409 событиями имеет 0 активных полей | DB query + summary.json |
| 13 | Идемпотентность сборки подтверждена builder (content_signature_equal=true в build logic) | build_tz16_profiles_v7.py + idempotence check |
| 14 | refresh_customer_profiles читает КОПИЮ journal (не live) | refresh.py строка 61-63 |

### ГИПОТЕЗА (не проверено по коду/данным)

| # | Гипотеза | Основание |
|---|---|---|
| H1 | Profile с 409 событиями — это технический/тестовый номер или автоответчик-спам | Косвенно: 0 полей при 409 событиях аномально |
| H2 | 70% unknown в mango_call = нет retrofit brand-тегов на звонках (retrofit скрипт только для TG/WA) | Читал retrofit_channel_brand_tags_in_timeline.py — он только для telegram/whatsapp |
| H3 | «Phantom слоты» (10+ child_key на профиль) в большинстве случаев = один ребёнок, о котором звонили многократно в разные периоды | Выглядит так по данным, но не верифицировал cross-call семантику |
| H4 | 317 профилей без телефона — contact-only записи (AMO-контакт без телефона) | Косвенно по структуре identity_links |

---

## Сводка ключевых чисел (v7, 2026-06-12)

- **Профилей:** 18 399
- **Активных полей:** 117 823 (из 190 614 всего)
- **Superseded:** 72 791 (38%)
- **brand:unknown:** 81 332 (69% активных)
- **brand:unpk:** 20 128 (17%)
- **brand:foton:** 15 528 (13%)
- **Merge-маркеров:** 835 (профилей с маркером: 824)
- **Профилей с 2+ child_slots до merge:** 4 589 → после: 4 411 (слито 178 пар)
- **Полей rekeyed при merge:** 4 593
- **Unmatched calls:** 1 053
- **Профилей без телефона:** 317 (1.7%)
- **child_name c ФИО (2+ слов):** 3 671 записей
