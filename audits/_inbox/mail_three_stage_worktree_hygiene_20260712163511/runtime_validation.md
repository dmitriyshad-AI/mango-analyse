# Ручная проверка на тестовой копии

Ревизия: `b39470964236556b6b381246c166c4596f5e4460`.

Источник Timeline открыт только на чтение и скопирован через SQLite backup в
`.codex_local/staging/customer_timeline_staging.sqlite`.

Три полных цикла сохранены локально в
`.codex_local/staging/mail_pipeline/cycles/final_cycle_{1,2,3}`:

- цикл 1: новых писем 0, строк обработки 0, курсор не изменился;
- цикл 2: одно новое отправленное письмо, импортировано один раз;
- цикл 3: одно новое входящее письмо, импортировано один раз; overlap дал две
  строки, но enrich обработал только одно новое событие.

Во всех циклах: download/process/import `status=ok`, `errors=0`,
`truncated=false`, `visibility_changed=false`.

Итог тестовой БД: `quick_check=ok`, FK violations `0`, необработанных pending
без причины `0`. Количество mail chunks с `allowed_for_bot=1` не изменилось.

AMO/Tallanto/CRM/client write: `0`. Боевая Timeline write: `0`.
LaunchAgent для трёх стадий не установлен.
