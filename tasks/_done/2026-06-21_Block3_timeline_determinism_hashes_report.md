# Block3 — timeline determinism hashes

Дата: 2026-06-20/21  
Режим: локально, без live-write и без внешних сетей.

## Статус

`PARTIAL`.

Строгий пункт ТЗ «пересобрать дважды на одних источниках и сравнить все таблицы» ночью не выполнен: Block1 с channel import занял существенно больше ожидаемого из-за FTS-синхронизации в Telegram/WhatsApp importers. Запуск второй полной сборки с каналами заведомо вышел бы за 20-минутный тайм-бокс Block3.

## Что сделано

Для новой БД из Block1 посчитаны детерминированные row-hashes всех таблиц:

`/Users/dmitrijfabarisov/Projects/Mango_tz139_customer_timeline/product_data/customer_timeline/canonical_readonly_20260621_with_channels/table_row_hashes_current.json`

БД:

`/Users/dmitrijfabarisov/Projects/Mango_tz139_customer_timeline/product_data/customer_timeline/canonical_readonly_20260621_with_channels/customer_timeline.sqlite`

Время расчёта hash-аудита: 23.373 сек.

## Вывод

Есть baseline row-hashes для утреннего регрейда/повторной сборки. Полная проверка `run_a == run_b` не заявляется как пройденная.

## Риск / следующий шаг

Главный технический блокер детерминизма ночью: channel importers пишут FTS построчно. Для практичной двойной пересборки нужно вынести deferred FTS/rebuild в Telegram/WhatsApp importers или сделать отдельный быстрый режим канального импорта.
