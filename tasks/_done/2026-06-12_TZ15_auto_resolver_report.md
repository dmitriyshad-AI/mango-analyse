# TZ-15 auto resolver report

Дата: 2026-06-12 15:24 MSK.

## Что сделано

- Реализован `DRAFT_LOOP_AUTO_RESOLVER` default OFF.
- Ручные и авто-пары поддерживают `not_before_ts`; старые сообщения и `timestamp=0` пропускаются с журналом.
- Авто-пары хранятся отдельным файлом, перечитываются на каждом цикле, повторный резолв не двигает watermark.
- Telegram auto-resolve: только точный числовой `Telegram ID`, ровно один контакт, ровно одна открытая сделка.
- Max auto-resolve: только телефон из полей Wappi, не из текста и не из `chat_id`; при недоступном stoplist общих номеров Max уходит в ручную привязку.
- `403`/allowlist-desync по одной сделке переводит пару в карантин и не останавливает весь цикл.
- Добавлен offline `--retro-report`: пишет только файл вне репозитория, в AMO не пишет.

## Немедленные ручные пары

- Две пары внесены в `~/.mango_secrets/draft_loop_pairs.json` с `not_before_ts`.
- Перед изменением сделаны бэкапы:
  - `~/.mango_secrets/draft_loop_pairs.json.bak_tz15_1781265476`
  - `~/.mango_secrets/amo_wappi_phase1.json.bak_tz15_1781265476`
- Старые входящие по этим парам вручную помечены processed в `~/.mango_local/draft_loop/state.json`, потому что на момент внесения пары код `not_before_ts` ещё не был в контуре.

## Блокер live-write по новым парам

- Серверный allowlist AI Office не обновлён: SSH на сервер вернул `Permission denied`.
- Live draft-loop процесс остановлен; стоп-файлы сейчас активны:
  - `~/.mango_secrets/STOP_DRAFT_LOOP`
  - `~/.mango_local/draft_loop/STOP_DRAFT_LOOP`
- Новый код защитит контур карантином при `403`, но примечания по новым сделкам не появятся, пока серверный allowlist не пополнен вручную.

## Max phone check

- Foton Max: в выборке Wappi телефон есть у большинства чатов.
- UNPK Max: телефон есть не у всех чатов.
- Политика реализации: нет телефона в полях Wappi или нет stoplist общих номеров -> ручная привязка, без обходов.

## Сухой прогон auto-resolver

Команда: `--once --dry-run`, пустой файл ручных пар, `DRAFT_LOOP_AUTO_RESOLVER=1`, `--chat-limit 3`.

Итог:

- `auth_error=false`
- `bot_calls=0`
- `matched=4`
- `closed_lead=1`
- `max_phone_missing=3`
- `shared_phone_stoplist_unavailable=1`

Персональные сырые данные dry-run лежали во временной директории вне репозитория.

## Retro report

- Файл: `~/.mango_local/draft_loop_inventory/retro_compare_20260612T122243Z.json`
- `rows=3`
- `bot_calls=3`
- AMO-write не выполнялся.

## Тесты

- Точечные: `31 passed`.
- Полный pytest: `3047 passed, 2 skipped, 1 warning`.

## Остаточные решения

1. Нужно отдельно обновить server-side allowlist AI Office для новых сделок и только после этого снимать стоп-файлы/перезапускать live-loop.
2. Для Max auto-resolve нужен утверждённый файл stoplist общих семейных номеров; пока его нет, Max безопасно уходит в ручные пары.
3. Авто-пары, найденные dry-run, требуют сверки архитектором перед server allowlist и live-write.
