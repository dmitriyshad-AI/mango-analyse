# Track 6 — Сводка неэффективности кода: слой CRM/клиенты
Дата аудита: 2026-06-13. Аудитор: субагент read-only.

---

## A. Инвентарь файлов слоя

| Модуль | Файлов .py | Строк всего |
|---|---|---|
| amocrm_runtime/ | 18 | 7 080 |
| deal_aware/ | 7 | 5 441 |
| customer_timeline/ | 18 | 14 925 |
| quality/ | 28 | 13 157 |
| **Итого src** | **71** | **40 603** |
| scripts/ (amo/profile/history/tallanto/card) | 42 из 293 | — |

Наиболее крупные файлы:
- `customer_timeline/store.py` — 2 174 стр.
- `customer_timeline/canonical_readonly_import.py` — 1 445 стр.
- `deal_aware/deal_text_builder.py` — 1 588 стр.
- `amocrm_runtime/deals.py` — 1 578 стр.
- `amocrm_runtime/amo_integration.py` — 1 286 стр.

---

## B. Дубли логики [ПРОВЕРЕНО]

### B1. `normalize_phone` — 4 независимые реализации (ВЫСОКИЙ приоритет)

Функция написана заново в четырёх местах с незначительными расхождениями:

| Файл | Строка | Поведение |
|---|---|---|
| `src/mango_mvp/utils/phone.py` | 7 | Канонная: +7, 8→7, 10 цифр→+7; возвращает `Optional[str]` |
| `src/mango_mvp/channels/telegram_history.py` | 794 | Та же логика, но без `len==10` ветки; возвращает `Optional[str]` |
| `src/mango_mvp/insights/phone_identity.py` | 10 | Расширенная (int'л номера 10-15 цифр); возвращает digit-only без `+` |
| `src/mango_mvp/productization/mail_archive.py` | 473 | Возвращает `str` (не Optional); формат `+7XXXXXXXXXX` |

`telegram_history.py` использует свою версию внутри `__post_init__` (строки 111, 146, 210), хотя `utils.phone.normalize_phone` уже импортирован в `phone_context.py` (строка 10). Разные реализации дают разные ключи → ошибки матчинга по телефону.

### B2. `safe_text` / `_safe_text` — более 25 независимых копий (ВЫСОКИЙ приоритет)

grep показал 27 определений `def safe_text` или `def _safe_text` по всему слою. Все делают одно: `str(value).strip()` с защитой от None. Примеры в зоне аудита:

| Файл | Строка | Реализация |
|---|---|---|
| `amocrm_runtime/deals.py` | 151 | `str(value).strip()` |
| `amocrm_runtime/phone_context.py` | 50 | `str(value).strip()` |
| `amocrm_runtime/agent_runtime.py` | 68 | аналог |
| `deal_aware/stage1_snapshot.py` | 870 | `re.sub(r"\s+", " ", str(value)).strip()` (нормализует пробелы) |
| `customer_timeline/context_provider.py` | 442 | аналог |
| `customer_timeline/canonical_readonly_import.py` | 1430 | аналог |
| `customer_timeline/deal_aware_sample_import.py` | 1030 | аналог |
| `customer_timeline/contact_control_sample_import.py` | 1256 | аналог |

Нет единой утилиты — каждый модуль варит собственную. Часть вариантов нормализует пробелы, часть нет.

### B3. `load_json` — 7 одинаковых копий в `deal_aware/` (СРЕДНИЙ)

Полностью идентичный код (файл→dict, пустой→`{}`, JSONDecodeError→`{}`):

- `deal_aware/stage1_snapshot.py:829`
- `deal_aware/deal_quality_gate.py:595`
- `deal_aware/deal_attribution.py:622`
- `deal_aware/deal_state_classifier.py:628`
- `deal_aware/deal_text_builder.py:1374`
- `deal_aware/deal_writeback.py:475`
- `question_catalog/answer_review_pack.py:131`

### B4. `write_sqlite` — 6 одинаковых копий в `deal_aware/` (СРЕДНИЙ)

Полностью идентичный код (создать таблицы, executemany):

- `deal_aware/stage1_snapshot.py:753`
- `deal_aware/deal_quality_gate.py:629`
- `deal_aware/deal_attribution.py:639`
- `deal_aware/deal_state_classifier.py:571`
- `deal_aware/deal_text_builder.py:1570`
- `deal_aware/deal_writeback.py:502`

### B5. `write_csv` / `_write_csv` — более 19 реализаций (СРЕДНИЙ)

В зоне аудита не менее 19 независимых определений `_write_csv` / `write_csv` с minor расхождениями (fieldnames из rows[0] vs sorted union, empty→write-empty vs skip).

### B6. `read_csv` — 6 независимых реализаций (СРЕДНИЙ)

- `deal_aware/stage1_snapshot.py:792` — `utf-8-sig`, возвращает `list[dict[str,str]]`
- `question_catalog/answer_review_pack.py:115` — аналог
- `question_catalog/rop_questionnaire.py:112` — аналог
- `insights/knowledge_base.py:1177` — аналог (`list[dict[str,Any]]`)
- `insights/rop_validation_pack.py:447` — аналог
- `knowledge_base/manager_answer_playbook.py:890` — аналог

### B7. `build_summary` — 10 функций с одинаковым именем (НИЗКИЙ)

10 функций `build_summary` в разных модулях слоя — не копии (разные сигнатуры), но затрудняют навигацию и grepping.

### B8. Дубль-скрипт: единственное/множественное число (СРЕДНИЙ)

`scripts/prepare_message_archive_history_full_cycle.py` (307 стр.) и  
`scripts/prepare_message_archives_history_full_cycle.py` (338 стр.) — 90%+ кода идентично.  
Разница: второй принимает несколько `--archive-dir`, добавляет `archive_calls_without_phone.csv`, удаляет `_safe_scalar`. Первый выглядит предшественником второго — **гипотеза: оригинал мёртв**, второй его заменяет.

---

## C. Тяжёлые операции в цикле [ПРОВЕРЕНО]

### C1. N+1 API-вызовы `fetch_lead` в цикле (ВЫСОКИЙ)

**Файл:** `amocrm_runtime/deals.py`, строки 934–941  
```python
for contact in contacts:
    ...
    for lead_id in lead_ids:
        leads.append(fetch_lead(session, lead_id=lead_id, ...))  # HTTP-запрос на каждый lead_id
```
На каждый контакт делается по одному HTTP GET /leads/{id} на каждую сделку. При 5 контактах × 3 сделки = 15 последовательных запросов. AMO API поддерживает batch-fetch через `GET /leads?ids[]=...`, но он не используется.

### C2. N+7 HTTP-запросов к Tallanto на каждый контакт (ВЫСОКИЙ)

**Файл:** `amocrm_runtime/tallanto_api.py`, строки 554–574  
`build_contact_context()` обходит `contacts` в цикле и для каждого делает 7 независимых запросов:
- `class_relations_by_contact`
- `opportunities_by_contact`
- `requests_by_contact`
- `finances_by_contact`
- `course_relations_by_contact`
- `abonements_by_contact`
- `classes_by_ids` (ещё цикл по class_ids внутри, строки 529–543: отдельный GET на каждый class_id)

При `max_contacts=10` → до 70+ запросов + classes N+1 сверху. Нет batch-API.

### C3. classes_by_ids — N+1 по class_id (ВЫСОКИЙ)

**Файл:** `amocrm_runtime/tallanto_api.py`, строки 529–543  
```python
for class_id in class_ids:
    payload = self.get_entry_by_id(module="most_class", entry_id=value)  # HTTP на каждый ID
```
Tallanto API не имеет bulk-endpoint для most_class — **гипотеза** (проверить по документации), но даже при его наличии код этого не использует.

---

## D. Многократное чтение одних данных [ПРОВЕРЕНО / ГИПОТЕЗА]

### D1. phone_context.py — файловый кеш с правильной инвалидацией (НИЗКИЙ, не проблема)

`amocrm_runtime/phone_context.py:109–128` реализует кеш с проверкой mtime. CSV читается только при изменении файла. Норма.

### D2. deal_text_builder.py читает stage1-файлы отдельно от deal_writeback.py (СРЕДНИЙ, ГИПОТЕЗА)

`deal_text_builder.py:137–142` читает `call_snapshot.csv`, `phone_rollup.csv`, `tallanto_students_snapshot.csv`, `writeoff_summary` из stage1_snapshot_root.  
Когда вызывается pipeline build_stage4 → build_stage5, эти же файлы читаются повторно через `deal_writeback.py:44–46` (другие файлы, но из той же директории). Нет shared in-memory cache между этапами.  
**Гипотеза:** при pipeline-прогоне одни и те же CSV читаются 2–3 раза разными стадиями.

### D3. canonical_readonly_import.py — захардкоженный путь к снимку (СРЕДНИЙ, проверено)

`customer_timeline/canonical_readonly_import.py:329–332` содержит хардкод:
```python
amo_root = project_root / "stable_runtime" / "deal_aware_amo_live_snapshot_20260513_v2"
mail_root = project_root / "_external_handoffs" / "mail_archive_2026-05-12" / ...
```
Эти пути можно переопределить через `config`, но дефолт смотрит на снимок от 13.05.2026 — **мёртвый дефолт** (актуальный снимок живёт в другом месте, определяемом CURRENT_RUNTIME.json).

---

## E. Мёртвый код / заброшенное [ПРОВЕРЕНО / ГИПОТЕЗА]

### E1. Захардкоженная дата `analysis_date = "2026-05-13"` в 3 конфигах (ВЫСОКИЙ)

**Проверено:** одинаковая заморожённая дата в трёх файлах:
- `deal_aware/deal_quality_gate.py:84`
- `deal_aware/deal_text_builder.py:126`
- `deal_aware/deal_writeback.py:37`

Это default-значение в датаклассе. Если не переопределяется при запуске — вся разметка «устарела» относительно июня 2026. Скорее всего переопределяется через CLI, но **риск**: запуск без явного `--analysis-date` тихо проставит май.

### E2. Захардкоженный токен-стейдж от 13.05 (ВЫСОКИЙ)

**Файл:** `deal_aware/deal_writeback.py:389, 396`  
```python
token = "WRITE_AMO_DEAL_AWARE_STAGE20_20260513"
"stage_id": "deal_aware_stage20_20260513"
```
Заморожены в коде — при следующем прогоне нужно будет менять вручную.

### E3. Хардкод пути к снимку `kc_source_extract_20260513` (СРЕДНИЙ)

**Файл:** `question_catalog/builder.py:131`  
Дефолт-путь `.codex_local/kc_source_extract_20260513/texts` захардкожен. Если директории нет — тихо пропускается (гипотеза: builder просто даёт пустые fact_sources).

### E4. Скрипт `build_student_card_manual_review_pack.py` — нет вызовов снаружи (СРЕДНИЙ, ГИПОТЕЗА)

**Проверено grep:** единственное упоминание имени файла — в `docs/WORKING_BASES_STATUS_2026-05-13.md`. Ни один другой `.py` файл не импортирует и не вызывает этот скрипт. Скрипт ссылается на `quality/tenant_text_normalizer.py` который живой, но сам скрипт, вероятно, заброшен после стейджа 13.05.

### E5. Скрипт `prepare_message_archive_history_full_cycle.py` (единственное число) — вероятно мёртв (СРЕДНИЙ, ГИПОТЕЗА)

Практически полная копия `*_archives_*` (множественное число, 338 стр.) с расширенной функциональностью. Нет признаков вызова первого из других файлов. **Гипотеза**: оригинал вытеснен вторым.

### E6. `customer_timeline/approved_context_pack.py` — нет pyc для Python 3.12 (НИЗКИЙ, ГИПОТЕЗА)

В `amocrm_runtime/__pycache__/` есть `.cpython-310.pyc` и `.cpython-312.pyc` для большинства файлов, но `customer_timeline/__pycache__/` содержит только `cpython-310`. Возможно, модули timeline не запускались под Python 3.12 — **гипотеза**, не проблема сама по себе.

### E7. Хардкоженный путь к AMO-снимку в canonical_readonly_import (СРЕДНИЙ)

Уже отмечен в D3. Дополнительно: строка 329 указывает `deal_aware_amo_live_snapshot_20260513_v2` — второй снимок (v2) как дефолт, но это не CURRENT_RUNTIME. Если CI/CD не передаёт явный config — пайплайн читает старые данные.

---

## F. Ранжирование по влиянию

| Приоритет | Находка | Влияние |
|---|---|---|
| ВЫСОКИЙ | C1 — N+1 fetch_lead в deals.py:940 | Задержка на каждый звонок пропорционально числу сделок |
| ВЫСОКИЙ | C2 — N+7 запросов к Tallanto за контакт | Tallanto-контекст медленный, блокирует черновик |
| ВЫСОКИЙ | C3 — classes_by_ids N+1 внутри C2 | Усиливает C2 |
| ВЫСОКИЙ | B1 — 4 normalize_phone с разным поведением | Ошибки матчинга «тихие» — клиент не найден |
| ВЫСОКИЙ | E1 — заморожённая дата 2026-05-13 в 3 дефолтах | Тихая ошибка при запуске без явной даты |
| ВЫСОКИЙ | E2 — захардкоженный токен stage20_20260513 | Ломается при следующем прогоне |
| СРЕДНИЙ | B2 — 25+ копий safe_text | Технический долг, риск расхождений |
| СРЕДНИЙ | B3 — 7 копий load_json в deal_aware | Технический долг |
| СРЕДНИЙ | B4 — 6 копий write_sqlite в deal_aware | Технический долг |
| СРЕДНИЙ | B8 — дубль-скрипт archive/archives | Путаница, вероятно один мёртв |
| СРЕДНИЙ | E3 — хардкод kc_source_extract_20260513 | Тихий пустой ввод |
| СРЕДНИЙ | E4 — build_student_card мёртвый скрипт | Мусор |
| СРЕДНИЙ | D3 — хардкод AMO-снимка в canonical_readonly | Читает старые данные по дефолту |
| НИЗКИЙ | B7 — 10 build_summary | Только навигация |
| НИЗКИЙ | D1 — phone_context кеш | Норма |
| НИЗКИЙ | D2 — повторное чтение CSV между стадиями | Незначительно (batch-скрипты) |

---

## G. Проверено vs гипотеза

**ПРОВЕРЕНО по коду:**
- B1: все 4 normalize_phone прочитаны, расхождения зафиксированы
- B2–B6: grep по def + ручная проверка 4–6 реализаций
- C1: deals.py строка 940 — fetch_lead в `for lead_id in lead_ids:`
- C2: tallanto_api.py строки 554–574 — 7 вызовов в цикле
- C3: tallanto_api.py строки 529–543 — get_entry_by_id в цикле
- E1: три файла deal_aware прочитаны, дефолт 2026-05-13 подтверждён
- E2: deal_writeback.py строки 389, 396 прочитаны
- E4: grep по имени скрипта дал только docs-упоминание

**ГИПОТЕЗА (не верифицировано глубоко):**
- E5: первый скрипт мёртв — не grep'нут по всем возможным вызовам (Makefile, sh-скрипты, docs)
- C3: Tallanto API не имеет bulk most_class — документацию не читал
- D2: повторное чтение CSV между стадиями — цепочка вызовов не прослежена до конца
- E3: builder тихо пропускает отсутствующий путь — поведение не прочитано до конца
