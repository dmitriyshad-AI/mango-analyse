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

- `must_handoff_vs_route`: 174 match / 67 mismatch
- `must_handoff_vs_p0_signal`: 172 match / 69 mismatch

Эти mismatch не являются автоматическим FAIL самого shadow-слоя, но запрещают включать frame в active decision path без следующего этапа:

- ручная `expected_frame` gold-разметка;
- разбор mismatch-классов;
- отдельный fail-closed этап, где frame может только усиливать ручную проверку, но не понижать существующий P0/brand/fact guard.
