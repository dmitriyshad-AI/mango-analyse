> DONE 2026-07-22 22:34 | ветка codex/ai-employee-timeline | codex

> TAKE 2026-07-22 02:00 | ветка codex/ai-employee-timeline | codex

Ветка: codex/ai-employee-timeline
Зоны: src/mango_mvp/customer_timeline/, src/mango_mvp/productization/mail_archive.py, src/mango_mvp/integrations/draft_loop.py, scripts/, deploy/customer_timeline_daily_captures/, tests/, docs/, tasks/, audits/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest --collect-only -q tests/test_customer_timeline_manager_dossier.py tests/test_marathon2_f9_actuality.py tests/test_customer_timeline_nightly_service.py tests/test_customer_timeline_amo_incremental.py tests/test_customer_timeline_ingestion.py
Семантический-аудит: да

# Customer Timeline до полезного контура ИИ-сотрудника

Цель: одной последовательной колеёй сделать staging Customer Timeline честным,
свежим и полезным менеджеру, переиспользуя существующие импортёры.

## Порядок

1. Исправить метрику свежести: отдельно показывать курсор данных, время
   успешного импорта и максимальную дату события.
2. Подключить готовый read-only AMO incremental к staging-контуру как shadow,
   не делая его обязательным до трёх чистых циклов.
3. Нормализовать `Contacts 10.07.2026.xls` в структурный Tallanto-снимок без
   колонки `История общения`, импортировать только в staging и проверить
   идемпотентность и привязку личности.
4. Проверить и догрузить в staging готовые локальные почтовые данные и
   read-only Wappi history; ambiguous не склеивать.
5. Закрыть повторное журналирование `pair_missing` без потери сообщения и
   запретить автоматическую привязку письма при конфликте phone/email.
6. После гейта свежести собрать менеджерские списки с причиной, следующим
   шагом и доказательством.
7. Провести тесты, смысловой аудит и оформить один audit pack для регрейда.

## Жёсткие границы

- AMO, Tallanto, CRM, Telegram и Wappi: запись и отправка равны нулю.
- Боевая Customer Timeline не изменяется и не публикуется без отдельного
  подтверждения владельца после SQL-регрейда staging.
- Не запускать ASR, Resolve+Analyze и тяжёлые LLM-пакеты локально.
- Не склеивать ambiguous/unmatched и не создавать bot-safe данные из
  непроверенных источников.
- Не устанавливать launchd-задачи и не включать массовый auto-resolver.
- OWNER-GATE: не публиковать staging в prod, не включать теневую автономность и
  не строить/включать транспорт отправки клиенту без отдельного решения владельца.

## Приёмка

- Свежесть каждого источника показывает `cursor_at`, `imported_at` и
  `max_event_at` раздельно; будущая дата события не делает источник свежим.
- Tallanto 10.07 импортирован в staging без `История общения`; повторный импорт
  не создаёт дублей; strong/ambiguous/unmatched посчитаны.
- AMO, почта и Wappi прошли read-only/staging цикл с отчётом по fetched,
  inserted, duplicate, ambiguous и unmatched.
- Staging проходит SQLite `quick_check`; P0/бренд/ПДн/анти-выдумка не менялись.
- Менеджерский результат проверен по сырью и прошёл независимый смысловой аудит.
- Полный безопасный pytest зелёный либо каждое исключение классифицировано и
  вынесено в отдельный блокер без подгонки кода.

## СТОП

- Любая попытка GET неожиданно требует write-доступ или меняет внешнюю систему.
- Источник недоступен/нестабилен после штатного backoff.
- Нет безопасной staging-БД либо путь указывает на боевую Timeline.
- Массовая ambiguous-привязка, потеря ранее доступных bot-safe chunks или
  SQLite `quick_check` не равен `ok`.
- Для продолжения требуется включить live-флаг, launchd или массовую отправку.
