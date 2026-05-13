# Аудит этапа: единый каталог вопросов клиентов

Дата: 2026-05-13

## Вердикт

Этап реализован и пригоден как безопасная основа для будущей базы вопрос-ответ. Получился воспроизводимый read-only процесс: собрать вопросы из звонков, Telegram и почты, очистить их, объединить в классы, подготовить черновые шаблоны и отдать РОПу рабочую таблицу проверки.

До автоматической отправки ответов клиентам этот слой еще не готов. Это правильно: текущая цель этапа - собрать и структурировать вопросы, а не дать боту право отвечать без утверждения.

## Что проверено

Код:

- `src/mango_mvp/question_catalog/contracts.py`
- `src/mango_mvp/question_catalog/safety.py`
- `src/mango_mvp/question_catalog/normalization.py`
- `src/mango_mvp/question_catalog/extractors.py`
- `src/mango_mvp/question_catalog/builder.py`
- `scripts/build_customer_question_catalog.py`

Тесты:

- `tests/test_question_catalog_contracts.py`
- `tests/test_question_catalog_normalization.py`
- `tests/test_question_catalog_safety.py`
- `tests/test_question_catalog_extractors.py`
- `tests/test_question_catalog_builder.py`

Выходные файлы:

- `product_data/question_catalog/customer_question_items.jsonl`
- `product_data/question_catalog/customer_question_classes.csv`
- `product_data/question_catalog/customer_question_classes.xlsx`
- `product_data/question_catalog/answer_templates.csv`
- `product_data/question_catalog/fact_requirements.csv`
- `product_data/question_catalog/current_fact_source_registry.json`
- `product_data/question_catalog/rop_question_review_pack.xlsx`
- `product_data/question_catalog/unanswered_questions.csv`
- `product_data/question_catalog/source_coverage_report.md`
- `product_data/question_catalog/question_catalog_summary.json`

## Итоги сборки

| Метрика | Значение |
|---|---:|
| Отдельных вопросов и сигналов | 6907 |
| Классов вопросов | 1013 |
| Шаблонов ответов | 1013 |
| Классов с динамическими фактами | 741 |
| Источников фактов-кандидатов | 41 |

По источникам:

| Источник | Обработано строк | Извлечено вопросов |
|---|---:|---:|
| Звонки | 2726 | 2431 |
| Telegram | 13223 | 1683 |
| Почта | 16609 | 2793 |

По статусам классов:

| Статус | Количество |
|---|---:|
| `template_ready_needs_current_fact` | 591 |
| `manager_only` | 315 |
| `draft_answer_exists_needs_review` | 104 |
| `needs_rop_answer` | 3 |

## Что сделано хорошо

1. Источники сведены в одну модель. Звонки, Telegram и почта теперь дают одинаковые сущности вопроса.
2. Есть защита от опасного поведения. Процесс не пишет во внешние системы, не меняет runtime-базы, не запускает распознавание и не отправляет сообщения.
3. Введено разделение между шаблоном и фактом. Для цен, расписания, скидок, адресов, документов и программ бот не получает готовый ответ без актуального источника.
4. Есть файл для РОПа. Можно не читать сырой JSON, а работать с понятной таблицей.
5. Каталог воспроизводимый. Сборка запускается одной командой и покрыта тестами.

## Остаточные недочеты

1. Класс `общий вопрос` сокращен с 876 до 292 элементов. Это соответствует целевому коридору, но остаток все равно нужно доразбирать после топ-100.
2. Почтовый источник покрывает только локально выгруженные архивы. Если в почте есть более старые или еще не выгруженные письма, они не попали в каталог.
3. Текущие источники фактов только найдены как кандидаты. Они не утверждены РОПом и не помечены как пригодные для автоматического ответа.
4. Финальные ответы РОПа еще не заполнены: `approved_question_answers_draft.xlsx` является черновиком.
5. Вопросы по сложным личным ситуациям пока отправляются в `manager_only`; это безопасно, но снижает автономность бота.

## Проверки безопасности

Ограничения соблюдены:

- AMO, Tallanto, CRM не изменялись.
- Telegram и почта не использовались для отправки сообщений.
- ASR и Resolve/Analyze не запускались.
- `stable_runtime` не изменялся.
- Выходная папка находится в `product_data/question_catalog/`.
- Для публичных примеров используется очистка текста.

Дополнительный контроль перед использованием:

- прогнать поиск телефонов и email по текстовым результатам;
- открыть `rop_question_review_pack.xlsx` и проверить, что примеры не содержат персональных данных;
- не подключать ответы к живому боту до утверждения РОПом.

## Тесты

Команда:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_question_catalog_*.py
```

Ожидаемый результат текущего прохода:

```text
34 passed
```

## Следующий шаг

Следующий этап - передать РОПу два рабочих файла:

1. `rop_review_priority_top100.xlsx` - проверка самых полезных классов.
2. `approved_question_answers_draft.xlsx` - фиксация решения РОПа и финального текста.
3. Подключение утвержденных файлов цен, расписания и адресов.
4. Повторная сборка, где появятся первые `approved_for_bot=true`.
5. Подключение только утвержденных классов к предпросмотру ответа бота.
