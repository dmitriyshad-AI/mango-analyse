# Semantic review

- Это `formal_pass` replay-конвейера, не semantic-pass реального Wappi экзамена.
- Бренды: machine gate проверяет чужой бренд в bot text; реальный replay должен отдельно считать brand violations по scrubbed диалогам.
- Цены/числа: machine gate проверяет новые числа против client-safe/prefix index. Полнота client-safe index будет проверяться на pilot-10.
- P0/деньги: machine gate требует менеджерский route и safety flags на P0 cases, но качество классификации P0 в реальном replay зависит от корректной разметки case.expected_p0.
- ПДн: pseudonymizer и `pii_signals` покрыты unit-тестами. Перед полным M1 нужен auto-grep по scrubbed pilot-10 и ручной просмотр 20 сообщений.
- Остаточный риск: manager reply не является абсолютным gold; сегмент `manager_issue_private` отделён, но требует ручного просмотра на методическом регрейде.
