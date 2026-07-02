# Semantic Review

Verdict: `PASS_WITH_NOTES`

## What Passed

- Отчёт не содержит клиентских текстов в Markdown: только dialog id, turn, route, action, confidence и причины.
- Отчёт явно разделяет два разных явления:
  - ручная метка `frame_too_cautious`;
  - настоящая ошибка поля `frame.must_handoff`.
- Реальная доминанта подтверждена: `semanticframe_existence_vs_availability=7`.
- Отчёт не выдаёт active-разрешение: `active_readiness=no_go`, `strict_active_candidates_now=0`, у каждой строки `active_allowed=false`.
- Опасные соседства вынесены отдельно: `danger_adjacent_do_not_lower`.

## Blocking Issues

Нет блокеров для report-only использования.

## Non-Blocking Risks

- Work item может попадать в несколько workstreams одновременно. Это осознанно: одна строка может одновременно требовать frame calibration, retrieval delivery и policy decision.
- Отчёт не решает саму проблему автономности. Он только задаёт порядок будущих ТЗ.
- Проверка exact proof опирается на существующий product existence catalog; перед active-этапом нужен отдельный paired eval.

## Required Gates Before Any Active Step

- `route/text diff=0` в shadow.
- `P0 lowered=0`.
- `too_confident=0`.
- `brand leaks=0`.
- `unsupported product/number/schedule claims=0`.
- Все active-кандидаты имеют fresh client-safe exact proof.
- Нет live availability, booking, payment, manager action или P0/money/legal/complaint соседства.
- Claude #1 semantic regrey + отдельное “да” Дмитрия.

## Recommended Next Action

Не включать F3. Следующий безопасный шаг — отдельный shadow-ТЗ на исправление SemanticFrame: различать “существует курс/формат” и “проверь места/запиши”, затем повторить M1 paired eval.
