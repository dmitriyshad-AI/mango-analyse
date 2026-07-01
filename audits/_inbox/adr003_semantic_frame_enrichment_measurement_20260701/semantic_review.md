# Semantic Review

## Вердикт

`formal_pass` для paired no-op измерителя.

`semantic_pass` для использования `SemanticFrame` как руля маршрута НЕ пройден.

## Почему

Измеритель корректно доказывает, что post-hoc frame metadata можно добавить без изменения готового черновика. Но frame не сравнен с ручной gold-разметкой, а локальные reports показывают расхождения с текущим route/P0-сигналом.

Wappi25:

- `must_handoff_vs_route`: 17 match / 8 mismatch
- `must_handoff_vs_p0_signal`: 13 match / 12 mismatch

Full131:

- `must_handoff_vs_route`: 203 match / 38 mismatch
- `must_handoff_vs_p0_signal`: 172 match / 69 mismatch

Эти mismatch не являются автоматическим FAIL самого shadow-слоя, но запрещают включать frame в active decision path без следующего этапа:

- ручная `expected_frame` gold-разметка;
- разбор mismatch-классов;
- отдельный fail-closed этап, где frame может только усиливать ручную проверку, но не понижать существующий P0/brand/fact guard.

## Gold Queue

Для следующего semantic-review шага создан builder очереди ручной разметки. По full131 он выделил 75 строк:

- 37: frame хочет handoff, но текущие P0-сигналы его не подтверждают;
- 32: frame хочет handoff при текущем self-route и без P0-сигнала;
- 3: frame хочет handoff при текущем self-route;
- 3: frame считает self-answer допустимым, но текущий route ведёт к менеджеру;
- брендовый баланс очереди: Foton 35 / UNPK 40.

Эта очередь нужна, чтобы отделить полезное снижение over-handoff от опасных пропусков. До заполнения `expected_*` gold и регрейда она не является разрешением включать frame как руль поведения.
