# Risk Review

Основные риски:

- `SemanticFrame` всё ещё не является самостоятельным decision policy: Ф3 active запрещена до Ф2 shadow и регрейда.
- Gold-75 покрывает mismatch/manager-action очередь, но не весь реальный поток; требуется class-specific shadow по каждому SELF-классу.
- `requested_action` accuracy остаётся ниже целевого ориентировочного 90% (`62/75`), поэтому action-зависимые решения нельзя включать.
- Confidence bucket `0.80-0.89` слабее (`must_handoff_accuracy=0.6364`); для следующего shadow-гейта стартовый порог должен быть не ниже `0.90`, пока Claude #1/Дмитрий не утвердят другой.

Защитные границы:

- Все изменения касаются только shadow prompt/parser и тестов.
- Default-OFF флаги не включались, профиль/live не трогались.
- P0 floor/preblock не менялись.
- Regex moratorium guard прошёл; новые regex-понимания не добавлялись.

