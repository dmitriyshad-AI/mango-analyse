> DONE 2026-08-04 22:01 | ветка main | codex

> TAKE 2026-08-04 21:39 | ветка main | codex

Ветка: main
Зоны: src/mango_mvp/customer_timeline/tallanto_attendance_import.py, tests/test_customer_timeline_tallanto_attendance_import.py, tasks/, audits/_inbox/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_customer_timeline_tallanto_attendance_import.py tests/test_customer_timeline_store.py tests/test_block4_wappi_personal_memory.py
Семантический-аудит: да

# Точный владелец Tallanto для посещений без ослабления семейного гейта

## Образ результата и бизнес-польза

Подтверждённое посещение или списание Tallanto попадает к тому ученику, чей
точный Tallanto student ID указан в отношении. Общий телефон семьи не скрывает
это событие. Настоящая коллизия нескольких клиентов остаётся закрытой, поэтому
данные одного ребёнка не попадают другому.

## Зафиксированная исходная точка

- код: `5b444302cd8f12a1e7c14942b531b230ad24063f`;
- staging DB: `customer_timeline_staging.sqlite`, размер `12078415872`,
  mtime `2026-08-04T14:34:20+0300`, SHA-256
  `df91aa9e6a6671072e22069de6ebca15524c19f2a218c356e478f6812c081420`;
- локальный replay: `~/.mango_local/customer_timeline_audits/
  attendance_unresolved_20260804/replay.json`, SHA-256
  `c871b3f0a8417a58a6398821a40ec05834a744d34a5b8af350d8b16ced884cfb`;
- manifest: тот же каталог, `manifest.json`, SHA-256
  `7acc17d78b5960abe65ef11affcea325ebb8422285813af6687d8a19a0122de3`;
- режим проверки: SQLite `-readonly`, без API и без записи;
- факт: у 16/16 один strong/manual-владелец точного Tallanto ID; для 15
  максимальное число прямых кандидатов `0..1`, для одного — `4`.

## Корень

`_load_unique_identity_customers()` использует customer-wide
`has_open_conflict`. Поэтому `shared_family_phone` или старый контактный
`ambiguous_identity` закрывает даже точный student ID. Название конфликта и
прямая ссылка на ID сами по себе недостаточны: у безопасных U06/U09/U10 есть
прямой `tallanto_identity_ambiguous`, но один кандидат; у опасного U15 — четыре.

## Перед кодом

1. Запустить `scripts/skills/inventory_before_build.py` по символам
   `_load_tallanto_customers,_load_unique_identity_customers,
   authoritative_exact_identity_rows`.
2. Подтвердить Graphify-карту на текущем HEAD и перечитать сырой код.
3. Не строить второй resolver, очередь или классификатор.

## Минимальная реализация

Использовать существующую `_load_tallanto_customers()` и существующие
`authoritative_exact_identity_rows()`/`timeline_conflicts`:

1. Для `tallanto_student_id` принять strong/manual-ссылку только при
   `owner_count=1`.
2. Customer-wide `has_open_conflict` не применять к точному attendance-ID.
3. Заблокировать ID, если есть открытый прямой `tallanto_identity_conflict`.
4. Открытый прямой `tallanto_identity_ambiguous` блокирует при
   `candidate_customer_count>1`; отсутствующее/невалидное число трактуется
   fail-closed. При числе `1` точный владелец допустим.
5. `ambiguous_identity` и `shared_family_phone` не закрывать и не менять: они
   продолжают блокировать сопоставление по телефону/email, но не exact-ID
   attendance.
6. API-путь attendance должен переиспользовать этот Tallanto-map. Проверка
   несовместимого точного AMO-владельца и подтверждённой семьи остаётся как есть.

Почему это минимум: меняется один существующий владелец Tallanto-map и один его
вызов; общий store, family graph, память и конфликтный реестр не ослабляются.
Бюджет: не более 50 добавленных строк нетестового кода, новых файлов, флагов,
зависимостей и LLM-вызовов — 0.

## Не делать в этой волне

- не менять `store.py` и `BLOCKING_FAMILY_CONFLICT_TYPES`;
- не закрывать массово старые конфликты;
- не удалять `_fetch_contact_rows`: он различает отсутствие контакта и
  инфраструктурный пробел;
- не обращаться к Tallanto/AMO/Wappi и не писать в staging/prod;
- не добавлять адресный pipeline ради одного replay.

## СТОП

Остановиться без правок, если replay не воспроизводит `15/1`, требуется менять
общий family/store gate, нет способа отличить одного кандидата от нескольких
структурными полями или дифф превышает бюджет 50 строк нетестового кода.

## Тесты, которые обязаны сначала краснеть

1. Уникальный exact student ID + открытые `shared_family_phone` и
   `ambiguous_identity` на его клиенте: attendance создаётся.
2. Прямой `tallanto_identity_ambiguous` с
   `candidate_customer_count=1`: attendance создаётся.
3. Тот же конфликт с `candidate_customer_count=4`: событие не создаётся,
   курсор не продвигается, статус `partial`.
4. Два strong/manual-владельца одного student ID: событие не создаётся.
5. Несовместимые Tallanto/AMO-владельцы: существующий NEG остаётся зелёным.
6. Мутация новой развилки в памяти обязана сделать тесты 1 или 3 красными.

## Приёмка

- целевые и соседние тесты зелёные;
- read-only расчёт на замороженном replay: `allowed=15`, `blocked=1`;
- общий family/store gate не изменён;
- `git diff --stat`: нетестовый код укладывается в бюджет и весь блок в сумме
  удаляет либо добавляет не более необходимого минимума;
- formal_pass, data_pass, semantic_pass и breaker_pass оформлены в одном audit
  pack; никаких live/runtime-записей не было.
