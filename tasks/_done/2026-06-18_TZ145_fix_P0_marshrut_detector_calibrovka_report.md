# ТЗ-145: калибровка детерминированного P0-детектора маршрута

Дата: 2026-06-18  
Ветка: `codex/tz145-p0-detector`  
Кодовый коммит для M1-бандла: `45ee45c65f749554414e19fba45f5866001b30a1`

## Что изменено

- `src/mango_mvp/channels/p0_recall_spec.py`
  - `_PAYMENT_BLOCK_GAP` больше не режет связку по дефису/тире/двоеточию/точке с запятой; разрыв только `.`, `!`, `?`, перевод строки.
  - `_PAYMENT_MOVED_PATTERN` расширен на `внесли/внесла/оплачен...`.
  - `_PAYMENT_RESULT_TARGET_PATTERN` расширен на `ссылка`, `приглашение`, `логин`, `пароль`, `платформа`.
  - missing-паттерн расширен на `не приш...`, `не дали`, `не дают`.
  - добавлен `QUALITY_COMPLAINT_RE` для жалоб на качество преподавания и подключён в `has_complaint_signal`.
- `src/mango_mvp/channels/dialogue_memory.py`
  - `HARD_P0_LATCH_CODES` расширен на `refund` и `complaint`.
  - `_suppress_presale_refund_latch` не менялся.
- `product_data/telegram_dynamic_test_sets/p0_stability_set_20260617.jsonl`
  - набор расширен с 10 до 20 строк: дефис-якорь, внесли/приглашение, ссылка/логин/пароль, complaint-качество, closing follow-up, benign-контроли.
- Тесты:
  - добавлены регрессии для payment/no-access и quality complaint;
  - старые ожидания по автоотпусканию `refund/complaint` обновлены на новый hard-latch контракт.

## Проверки

- JSONL валиден: 20 строк, `dialog_id` уникальны.
- Точечные regex-проверки:
  - `Оплатил онлайн-математику — доступа так и нет.` → `payment_dispute`
  - `Внесли оплату, приглашение не пришло.` → `payment_dispute`
  - `Оплатили курс, логин и пароль не дали.` → `payment_dispute`
  - `Оплатил, ссылка на платформу не пришла.` → `payment_dispute`
  - `Преподаватель не объясняет, ребёнок ничего не понимает.` → `complaint`
  - `Как выбрать преподавателя?`, `Оплачу позже.`, `Оплатил два курса.` → не P0
- Точечные тесты: `157 passed`.
- Полный pytest: `3351 passed, 5 skipped, 1 warning`.

Команда полного pytest:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q
```

## M1-бандл

Бандл собран, M1-прогон не запускался.

- `bundle_id`: `mango_clean_45ee45c6`
- путь: `/Users/dmitrijfabarisov/Yandex.Disk.localized/OpenClaw/Actual Mango Tests/mango_clean_45ee45c6`
- `manifest.json`: есть
- `BUNDLE_INFO.txt`: есть
- `head`: `45ee45c65f749554414e19fba45f5866001b30a1`
- KB snapshot: `product_data/knowledge_base/kb_release_20260612_v6_7_staging_r4_1/kb_release_v3_snapshot.json`
- расширенный `p0_stability_set_20260617.jsonl` есть внутри бандла.

## Смысловой вывод

`formal_pass`: да, тесты зелёные.  
`semantic_pass`: локально PASS_WITH_NOTES: P0-пол остался детерминированным, presale-защита не ослаблена, benign-контроли не форсятся в P0. Окончательный регрейд по сырью и динамическим транскриптам — за Claude/Дмитрием.

## Не делалось

- M1-прогон не запускался.
- Модель/API-ключи не использовались.
- CRM/AMO/Tallanto не трогались.
- `answer_safety_classifier.py::_suppress_presale_refund_latch` не менялся.
