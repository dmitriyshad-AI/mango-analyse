> DONE 2026-08-04 02:29 | ветка codex/noncontentful-call-memory-integration-20260804 | codex

> TAKE 2026-08-04 01:01 | ветка codex/noncontentful-call-memory-integration-20260804 | codex

Ветка: codex/noncontentful-call-memory-integration-20260804
Зоны: src/mango_mvp/customer_timeline/source_policy.py, src/mango_mvp/customer_timeline/bot_safe_runtime_context.py, src/mango_mvp/customer_timeline/read_api.py, src/mango_mvp/customer_timeline/store.py, src/mango_mvp/customer_timeline/stage4b_bot_opening.py, src/mango_mvp/customer_timeline/stage3_maintenance.py, src/mango_mvp/customer_timeline/safety.py, tests/test_customer_timeline_ingestion.py, tests/test_bot_safe_runtime_context.py, tests/test_customer_timeline_read_api.py, tests/test_customer_timeline_stage4b_bot_opening.py, tests/test_customer_timeline_stage3_maintenance.py, tests/test_customer_timeline_contracts.py, docs/worktrees_registry.md, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_customer_timeline_ingestion.py tests/test_bot_safe_runtime_context.py tests/test_customer_timeline_read_api.py tests/test_customer_timeline_stage4b_bot_opening.py tests/test_customer_timeline_stage3_maintenance.py tests/test_customer_timeline_contracts.py tests/test_telegram_public_pilot_bots.py
Семантический-аудит: да

# Интеграция полезного смысла ветки пустых звонков

## Исходные факты

- Актуальный `main`: `42ecb4971e8e74306d1fba44d17aeabd9602f228`.
- Донор: `codex/customer-timeline-junk-map-20260731@ccff3f48`.
- Публичный Telegram-гейт донора уже поглощён `main` более строгой реализацией.
- Донорский `call_chunk_maintenance.py` только считает строки, использует опасный
  для WAL режим `immutable=1` и не даёт бизнес-эффекта.
- На опубликованной read-only базе от 17 июля: 59 043 видимых звонковых
  фрагмента, из них 19 765 структурно несодержательных; у 4 706 клиентов такой
  фрагмент последний, в проверенной реальной десятке он дошёл до промпта 10/10.
- На свежей staging-базе: 23 593 структурно несодержательных фрагмента и 0
  видимых. Значит массовая переоценка уже существует в `stage4b`.

## Бритва: три варианта

1. Слить донор целиком: отклонено, ослабляет Telegram-гейт и добавляет дубль.
2. Перенести только инвентаризатор: отклонено, бизнес-эффект нулевой.
3. Исправить канонический предикат и последний read-time путь: выбран, потому
   что решает проблему старой опубликованной базы без новой подсистемы и записи.

## Образ результата

Структурно подтверждённый пустой звонок не попадает в память и промпт даже при
устаревших `allowed_for_bot=1/requires_manager_review=0`. Содержательный звонок
того же клиента остаётся. История и строки Timeline не удаляются. Чужой клиент
не появляется. Публичный бот и внешние системы не затрагиваются.

## Реализация

1. Исправить `is_non_contentful_call_record()` для JSON `false` и числа `0`,
   сохранив единственный канонический структурный предикат. Противоречие между
   `contentful=Нет` и содержательным типом звонка не угадывать: реальный замер
   нашёл среди 2 750 таких записей и полезные разговоры, и пустые дозвоны.
   Они остаются закрытыми до отдельной офлайн-переоценки моделью.
2. В `_customer_call_bot_items()` повторно проверять связанное событие этим же
   предикатом до попадания фрагмента в результат. Не использовать текстовые
   слова `недозвон/сброс/автоответ` как доказательство мусора.
3. В существующем `stage4b` добавить явную финальную метрику/инвариант:
   открытых структурно несодержательных звонков после apply должно быть 0.
4. В `stage3` переиспользовать общий регистронезависимый writable-path guard и
   удалить локальный дублирующий `_reject_prod_path`.
5. Общий writable-path guard должен отвергать жёсткую ссылку на другой файл:
   запись в hardlink меняет исходный inode и нарушает запрет production-write.
6. Не добавлять новый модуль, флаг, зависимость или удаление данных.

## Приёмка

- Реальная замороженная десятка на опубликованной базе: пустая фраза в промпте
  `10/10 -> 0/10`.
- Синтетический сквозной сценарий: старый разрешённый пустой звонок скрыт,
  более старый содержательный звонок того же клиента виден.
- JSON `"Нет"`, `false`, `0` закрываются; `true` и нормальный технический
  разговор не закрываются.
- Противоречивая пара `contentful=Нет` плюс содержательный тип остаётся закрытой,
  чтобы детерминированная эвристика не открыла пустые дозвоны.
- `stage4b` закрывает ранее открытый пустой звонок, повторный проход меняет 0.
- `COUNT(*)` не меняется; `DELETE` нет; записи во внешние системы и клиентские
  отправки равны 0.
- Публичный Telegram-гейт из `main` остаётся без изменений и его тесты зелёные.
- Символическая, относительная и жёсткая ссылка не позволяют обойти запрет
  записи в production Timeline.
- Архитектор, ломатель и бизнес-аудитор отдельно дают PASS.

## СТОП

- Любая запись в опубликованную/production Timeline.
- Любая классификация мусора по ПДн, бренду, общему телефону или слабым словам.
- Любое изменение Telegram-отправки либо live-службы.
- Новый параллельный классификатор вместо `is_non_contentful_call_record()`.
