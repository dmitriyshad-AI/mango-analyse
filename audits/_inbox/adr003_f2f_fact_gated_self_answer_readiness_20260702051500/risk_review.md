# Risk Review

## Главный риск

Слишком рано включить active route demotion и начать понижать `manager_only`.

## Защита

F2f делает только report. В коде нет runtime-проводки.

Strict candidate требует одновременно:

- `route == draft_for_manager`;
- exact product proof;
- явно заполненный frame safe/self;
- no P0/money/danger.

`manager_only` всегда отдельная группа `manager_only_exact_proof_needs_policy`, не active.

## Остаточный риск

Следующий этап должен решать, почему эти кейсы становятся `manager_only`, а не `draft_for_manager`, и можно ли это исправить upstream без трогания P0 floor/preblock.
