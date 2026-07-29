> TAKE 2026-07-29 07:19 | ветка main | codex

Ветка: main
Зоны: AGENTS.md, docs/worktrees_registry.md, src/mango_mvp/customer_timeline/, src/mango_mvp/integrations/draft_loop.py, scripts/build_customer_timeline_nightly_dv2_sources.py, scripts/import_tallanto_payments_to_timeline.py, scripts/run_amo_wappi_draft_loop.py, scripts/run_customer_timeline_codex_task.py, tests/, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest --collect-only -q tests/test_customer_timeline_nightly_service.py tests/test_customer_timeline_codex_task.py tests/test_customer_timeline_tallanto_cards_sync.py tests/test_import_tallanto_payments_to_timeline.py
Семантический-аудит: да

# ТЗ: актуальная Customer Timeline на всех источниках

## Разрешённое исполнение

По прямому запросу владельца разрешены сетевые чтения AMO, Tallanto и Wappi и
запись только в существующую staging-базу под
`~/.mango_local/customer_timeline_nightly/.codex_local/staging/`.

Запрещены prod publish, запись в AMO/Tallanto/CRM/Wappi, отправка сообщений,
ASR, Analyze и создание второй копии базы.

## Приёмка: готово, когда

1. Старый PID признан stale; параллельного процесса нет.
2. Канонический runner пересобрал конфигурацию текущей версии и завершил один
   полный проход.
3. Все десять обязательных источников подтверждены в этом проходе, включая
   AMO, карточки/оплаты/посещения Tallanto и Telegram/MAX Wappi.
4. `quick_check=ok`, внешние ключи целы, семейный граф не пуст.
5. Второй проход на той же базе не создаёт дубли и не меняет уже нормализованные
   строки без нового источника.
6. Собраны свежие 30 семей и Owner50 для смыслового аудита; ложные личности,
   конфликты и непривязанные события разобраны по причинам.
7. Боевая база не публикуется без отдельного решения Дмитрия.
8. Обычный повторный проход не перечитывает полный источник: использует
   сохранённую staging-базу и получает только новое/изменённое с перекрытием.
9. Каждый сетевой шаг сообщает режим, число полученных/новых/изменённых/
   переиспользованных строк и время чтения/обработки/записи.
10. Полный проход остаётся редкой сверкой полноты и не маскирует сбой
    инкрементального режима автоматическим дорогим откатом.
11. Приёмка считает живую воронку `Wappi/AMO identity -> семья -> досье ->
    текст памяти в запросе бота`; число сохранённых chunks само по себе не
    считается доказательством ни пустой, ни полной памяти.
12. Обязательный источник подтверждается текущим шагом и свежестью исходных
    данных, а не старой строкой или одним свежим `updated_at` курсора.

## СТОП

- Любой required source partial/failed: не публиковать и не запускать слепой
  повтор; сначала исправить доказанную причину.
- Не ослаблять обязательные источники, точные ID и конфликтные гейты ради
  зелёного отчёта.
- Не создавать новую базу, если существующая staging-база исправна.
