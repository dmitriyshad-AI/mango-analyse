# CRM card request/blocks RU cleanup: 2026-06-20

Статус: formal_pass, semantic_pass = PASS_WITH_NOTES.

## Что исправлено

- `Запрос` теперь строится из данных ребёнка в разборе звонка:
  - `child_fio` + `grade_current` из `call_analysis.structured_fields/crm_blocks`;
  - интересы/продукты/формат/предметы без дублей;
  - имя родителя из колонки `Имя` больше не подставляется как ребёнок.
- `Следующий шаг` получает дату звонка-источника в формате `текст (от ДД.ММ.ГГГГ)`.
- `Последняя сводка`, `История общения`, `Tallanto` сохраняют plain-text переносы строк.
- `История общения` больше не ссылается на старое поле `Последняя AI-сводка` и не дублирует возражения/интересы, которые вынесены в отдельные поля.
- `Блокеры` и `Предупреждения` показывают русские фразы вместо внутренних кодов.
- `Tallanto` показывает человекочитаемый статус:
  - `exact_phone_single` → `Один ученик по телефону` + доступные детали;
  - `exact_phone_multiple` → `На телефоне несколько учеников — проверить вручную`;
  - `no_exact_phone_match` → `Точного совпадения по телефону в Tallanto нет`.

## Preview

Новая папка:

`/Users/dmitrijfabarisov/Projects/Mango_etap1_crm_cards/product_data/customer_timeline/crm_card_preview_20260621_request_blocks_ru_r1/`

Артефакты:

- `crm_cards_preview.xlsx`
- `crm_cards_preview.csv`
- `crm_cards_preview.summary.json`
- `build_stdout.json`

Сводка:

- rows: 50
- ready_yes: 18
- ready_no: 32
- blocker_counts:
  - `На телефоне несколько человек — проверьте, к кому относится`: 19
  - `Не найден контакт в AMO`: 10
  - `Не найдена сделка в AMO`: 21
- safety:
  - write_amo=false
  - write_tallanto=false
  - write_customer_timeline=false
  - live_network_calls=false

## Машинные проверки preview

- английские коды в видимых полях: 0;
- строки с маскированным телефоном: 0/50;
- строки, где `Имя` контакта попало в `Запрос`: 0/50;
- строки с дублями элементов `Запрос`: 0/50;
- `Следующий шаг` с датой источника: 34 строки;
- `Последняя сводка` с переносами строк: 44 строки;
- `История общения` с переносами строк: 44 строки;
- `Tallanto` с переносами строк: 12 строк.

## Тесты

Targeted:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_crm_card_aggregator.py tests/test_amo_writeback_guards.py
```

Результат: `42 passed in 1.21s`.

Full:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests
```

Результат: `3377 passed, 5 skipped, 1 warning in 48.50s`.

## Semantic review

Вердикт: PASS_WITH_NOTES.

Что стало лучше:

- менеджер видит ребёнка/класс/интерес вместо родительского имени в запросе;
- блокеры теперь объясняют действие по-русски;
- история стала читаемой plain-text структурой без HTML и Markdown;
- Tallanto не выглядит как технический статус.

Остаточный риск:

- preview построен на 50 строках, финальный смысловой регрейд по XLSX остаётся за архитектором;
- если в старом анализе звонка `child_fio` ошибочно содержит ФИО родителя, карточка покажет именно то, что в разборе; имя контакта из `Имя` больше не используется как fallback.
