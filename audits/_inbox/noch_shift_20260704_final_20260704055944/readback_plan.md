# Readback plan

Для будущего AMO write:

1. Перед записью выполнить anti-clobber/pre-patch gate по текущим значениям полей.
2. Записывать только 3 approved ready rows и только после owner approval.
3. После записи сделать AMO GET по contact/lead и сверить note/field ids, payload hash и отсутствие client send.
4. При любом `would overwrite non-empty unexpected` — стоп без записи.

Н6 текущей ночи не является readback после записи; это только readiness diff.
