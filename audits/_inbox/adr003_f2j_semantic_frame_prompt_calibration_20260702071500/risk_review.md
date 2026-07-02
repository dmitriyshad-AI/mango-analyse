# Risk Review

## Runtime риск

Низкий: изменён только prompt для shadow-флага, который default OFF и не включён в профиль.

## Live риск

Live Telegram/Wappi/AMO/CRM/Tallanto не трогались.

## Главный риск

Перепутать частичное улучшение `requested_action` с готовностью active. Это запрещено:

- `must_handoff_wrong` не улучшился;
- active remains NO-GO;
- route/text не менялись.

## Остаточный риск

Модель может на полном наборе повести себя иначе, чем на локальном subset. Нужен полный paired shadow eval.
