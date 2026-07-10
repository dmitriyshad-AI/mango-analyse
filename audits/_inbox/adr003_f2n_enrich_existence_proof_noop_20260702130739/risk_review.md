# Risk Review

## Риск

Низкий для runtime: изменён только измерительный harness
`scripts/run_telegram_dynamic_client_sim.py`.

## Проверки

- Runtime direct path уже имел F2l proof-shadow в provider.
- Enrichment path теперь вызывает тот же proof-shadow, но работает с frozen
  `SubscriptionDraftResult`.
- Тест проверяет, что route/text/safety/checklist остаются frozen.
- Paired отчёт подтвердил route/text/input diff = 0.

## Active-риск

Active остаётся NO-GO. В paired no-op замере нет self-answer candidates.
Manager-only/context-update строки с exact proof не понижать без отдельного
owner policy decision и нового shadow.
