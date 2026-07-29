> DONE 2026-07-29 06:03 | ветка main | codex

> TAKE 2026-07-29 05:52 | ветка main | codex

Ветка: main
Зоны: scripts/build_customer_timeline_nightly_dv2_sources.py, scripts/run_customer_timeline_codex_task.py, src/mango_mvp/customer_timeline/nightly_service.py, tests/test_customer_timeline_mail_link_enrich.py, tests/test_customer_timeline_nightly_service.py, tests/test_customer_timeline_codex_task.py, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_customer_timeline_mail_link_enrich.py tests/test_customer_timeline_nightly_service.py tests/test_customer_timeline_codex_task.py
Семантический-аудит: да

# ТЗ: повторно привязывать старые письма после обновления Tallanto

## Сырой факт

Механизм `reconsider_pending` уже существует, но ночной builder и parser его
не передают. Поэтому старые письма с `pending_reason` не пересматриваются даже
после появления точного телефона или email из свежей карточки Tallanto.

## Образ результата

1. Ночной процесс явно включает существующий `reconsider_pending=true`.
2. Parser передаёт значение в существующий `MailLinkEnrichConfig`.
3. Схема конфигурации повышена, старый JSON fail-closed отвергается.
4. Уже надёжно привязанное письмо не обрабатывается повторно.
5. Повторный одинаковый проход не создаёт изменений.

## СТОП

- не создавать второй классификатор, таблицу, флаг или зависимость;
- не читать тела реальных писем и не менять staging/prod;
- не отправлять письма и не писать во внешние системы.

## Приёмка: готово, когда

- целевые и полные CPU-тесты зелёные;
- отрицательный контроль без флага сохраняет старое pending;
- точные Tallanto email/phone снимают pending при включённом режиме;
- audit pack, один коммит, push в оба зеркала.
