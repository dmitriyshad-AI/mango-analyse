# Semantic Review

## Verdict

PASS_WITH_NOTES.

## What Passed

- Слой не обещает клиенту наличие мест или запись.
- Слой не делает active-понижение маршрута.
- Proof ограничен stable existence/format и fresh client-safe KB facts.
- P0, деньги, документы, запись, live availability и личный статус остаются
  вне proof-границы.

## Blocking Issues

Active включение по-прежнему заблокировано. Нужен полный paired shadow eval:

- `too_confident=0`;
- P0/money/manager_only lowered = 0;
- route/text diff = 0 для shadow;
- отдельная сверка брендов и выдумок.

## Non-Blocking Risks

- `product_existence_axes_catalog` может не покрывать часть реальных фактов,
  если в KB нет структурных осей.
- Proof-shadow не означает, что финальный черновик уже хорошо отвечает клиенту:
  это только доказательство, что стабильный факт существует.

## Required Regression/Gate

Перед active-гейтом требуется проверка, что self-answer использует только
свежий client-safe exact proof и не трогает mixed requests:
справка + запись/места/оплата/документы/личный статус.
