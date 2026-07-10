# ADR-003 F2f Fact-Gated Self-Answer Readiness

## Что сделано

Добавлен report-only scorer `scripts/report_adr003_fact_gated_self_answer_readiness.py`.

Он использует F2e existence report и отделяет:

- strict F3 candidates: `draft_for_manager` + exact product proof + safe/self frame;
- exact-proof `manager_only`, которые нельзя понижать Ф3 route gate;
- already-self exact proof;
- no-proof/danger/money/P0 blocked rows.

## Что это НЕ делает

- Не меняет route/text.
- Не подключается к direct path/provider/profile.
- Не включает флаги.
- Не понижает `manager_only`.

## Реальный пересчёт 36ea110

См. `real_36ea110/adr003_fact_gated_self_answer_readiness_report.md`:

- strict F3 draft candidates: 0;
- manager-only exact-proof needs policy: 2;
- already self exact proof: 6;
- blocked no exact proof: 1;
- excluded danger/money/P0: 1.

## Вывод

Ф3 active по железным правилам остаётся NO-GO: нет `draft_for_manager` exact-proof кандидатов.

Реальный рычаг есть, но он upstream/policy: 2 строки уже имеют exact KB-proof, но текущий route `manager_only`. Их нельзя трогать понижающим Ф3-гейтом без отдельного решения.

## Исправления после независимого аудита

- strict-кандидат теперь требует явно заполненный safe/self frame; пустой `risk_class`/`answerability` блокируется;
- F2f берёт полный `rows` из F2e report, а не обрезанные `examples[:50]`;
- добавлены регрессии на >50 строк, money-only без P0-флага и `not_offered` exact proof.
