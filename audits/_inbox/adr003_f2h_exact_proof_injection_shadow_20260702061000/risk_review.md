# Risk Review

## Главный риск

Перепутать freshness/evidence proof с разрешением на autonomous answer.

## Защита

F2h делает только report. Acceptance всегда `active_readiness=no_go`.

Residual blockers считаются отдельно от fact proof.

## Остаточный риск

Для реального повышения автономности нужен новый shadow-дизайн. Пока неясно, что лечить первым:

- доставку exact fact в runtime/retrieval;
- калибровку frame для `check_availability` vs справка о существовании;
- policy для `context_update`.

Каждый вариант требует отдельной приёмки Claude #1.
