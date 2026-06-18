# ТЗ-145.1: внутренний зазор access-слов в P0 payment detector

Дата: 2026-06-18  
Ветка: `codex/tz145-p0-detector`  
Кодовый коммит для M1-бандла: `0c3bce5d0170bd7db0f6e39ee2a1cbbf0d84d3e4`

## Что изменено

- В `src/mango_mvp/channels/p0_recall_spec.py` `_PAYMENT_RESULT_TARGET_PATTERN` разделён на два класса:
  - `ACCESS`: `ссылк`, `приглашени`, `логин`, `парол`, `платформ`, `доступ` с широким зазором `[^.!?\n]{0,60}`.
  - `GENERIC`: `платёж`, `оплата`, `занятия`, `курс`, `кабинет` со старым узким зазором `[^,.:;!?—–\-\n]{0,15}`.
- `_PAYMENT_RESULT_MISSING_PATTERN` собран из 4 веток: `ACCESS→MISS`, `MISS→ACCESS`, `GENERIC→MISS`, `MISS→GENERIC`.
- `MOVED`, `_PAYMENT_BLOCK_GAP`, duplicate-паттерн, latch и presale-suppress не менялись.
- `tests/test_answer_safety_classifier.py` расширен wide-gap access-регрессиями и benign-якорем.
- `product_data/telegram_dynamic_test_sets/p0_stability_set_20260617.jsonl` обновлён до 22 строк: разнесённые приглашение/ссылка/логин/доступ и benign «Оплатила вчера, занятия завтра — в системе пока нет».

## Проверки

- Таблицы `P0_TRUE_POSITIVE_CASES` / `P0_BENIGN_CASES`:
  - `TP 42/42`
  - `BENIGN false positives 0/26`
- Ручные regex-проверки:
  - `Оплату внесли, а приглашение на онлайн физику для 9 класса так и не пришло.` → `payment_dispute`
  - `Оплатили, ссылка на платформу для онлайн занятий до сих пор не пришла.` → `payment_dispute`
  - `Оплачен курс, логин и пароль в личном кабинете так и не дали.` → `payment_dispute`
  - `Оплатили курс, доступ к личному кабинету для онлайн физики не открыли.` → `payment_dispute`
  - `Оплатила вчера, занятия завтра — в системе пока нет.` → не P0
- JSONL: 22 строки, `dialog_id` уникальны.
- Точечные тесты: `162 passed`.
- Полный pytest: `3356 passed, 5 skipped, 1 warning`.

## M1-бандл

M1-прогон не запускался.

- `bundle_id`: `mango_clean_0c3bce5d`
- путь: `/Users/dmitrijfabarisov/Yandex.Disk.localized/OpenClaw/Actual Mango Tests/mango_clean_0c3bce5d`
- `head`: `0c3bce5d0170bd7db0f6e39ee2a1cbbf0d84d3e4`
- KB snapshot: `product_data/knowledge_base/kb_release_20260612_v6_7_staging_r4_1/kb_release_v3_snapshot.json`
- `p0_stability_set_20260617.jsonl` внутри бандла: 22 строки.

## Смысловой вывод

`formal_pass`: да.  
`semantic_pass`: локально PASS_WITH_NOTES. Широкий зазор теперь применяется только к access-словам, а generic-зазор остался узким, поэтому benign-якорь «занятия завтра — в системе пока нет» не стал P0. Финальный вердикт по динамическому сырью — за Claude/Дмитрием.

## Не делалось

- M1-прогон не запускался.
- Модель/API-ключи не использовались.
- CRM/AMO/Tallanto не трогались.
