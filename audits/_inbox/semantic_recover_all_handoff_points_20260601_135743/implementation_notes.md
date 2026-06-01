# Semantic recover all handoff points

Дата: 2026-06-01
Ветка: codex/m1-intermediate-20260529-final

## Что сделано

- Подтверждены чтением точки возврата ухода в `run_pipeline`: contract-manager-only, no-draft, draft-error, semantic unavailable, repair-fail/hard verification failed, coverage-repair.
- Подтверждены сигнатуры `_verified_empty_handoff_replacement`, `_has_exact_retrieved_answer_part`, `_hard_check`, `_coverage_cite_only_answer`.
- Добавлен общий cite-only recover перед безопасным handoff: если есть точный retrieved-факт той же темы, не P0 и не опасный соседний факт, бот отвечает сам текстом только из факта.
- Recover подключен ко всем найденным точкам возврата ухода, включая no-draft, draft-error, semantic-unavailable и hard-verification-failed.
- Соседние платежные факты больше не подаются вторичной справкой на вопрос про другой способ оплаты.
- Understanding prompt не трогался.

## Механика защиты

- Recover блокируется для P0, возврата, жалобы, юридических и платежных спорных сигналов.
- Recover разрешен только при точном покрытии `retrieved_facts` или явно разрешенном key-coverage.
- Лагерь/смена не заменяются фактом регулярного курса.
- Кандидат проходит `_hard_check`; при новых конкретных якорях без доступной семантики recover не применяется.

## Отклонения

- Вместо отдельного semantic-match в этих точках использован cite-only recover из retrieved-фактов. Это уже требовалось ТЗ как безопасный путь для случаев, где LLM-критик недоступен или repair сорвался.
