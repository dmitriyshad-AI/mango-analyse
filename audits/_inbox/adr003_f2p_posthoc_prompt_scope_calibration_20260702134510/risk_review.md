# Risk Review

## Что не менялось

- route/text живого ответа;
- профиль `pilot_gold_v1`;
- live bot;
- P0 floor/preblock;
- manager action gate;
- self-answer active policy.

## Основные риски

1. Модель может начать слишком смело считать справку answer_self.
   Контроль: gold calibration показал `too_confident=0`, `p0_misses=0`.

2. Route-only demotion может быть включён преждевременно.
   Контроль: отчёт сохраняет `strict_active_candidates_now=0` и
   `active_readiness=no_go`.

3. Фактическая справка может стать выдумкой без проверенного источника.
   Контроль: остаток помечен как `fact_assertion_required=8`; active запрещён
   до отдельного fact-verification слоя.

## Остаточный риск

Это не production-ready изменение поведения. Это только калибровка shadow-рамки
и отчётности. Для включения автономности нужен новый этап с проверенным fact path
и повторным регрейдом Claude #1.
