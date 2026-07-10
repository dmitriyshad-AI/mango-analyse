# Risk Review

## Основной риск

Отчёт могут неправильно прочитать как список кандидатов на self-answer.

Митигация:

- `active_readiness=no_go`;
- `strict_active_candidates_now=0`;
- каждая строка содержит `active_allowed=false`;
- каждая строка содержит `active_block_reason`.

## Runtime риск

Runtime отсутствует: скрипт только читает M1 transcripts, gold labels и KB snapshot.

## Live риск

Live Telegram/Wappi/AMO/CRM/Tallanto не трогались.

## Semantic риск

Главная ошибка, которую нельзя допустить: считать `check_availability` harmless reference. В отчёте это выделено как `semanticframe_existence_vs_availability`, а не как active-кандидат.

## Остаточный риск

Перед любым active-этапом нужна новая shadow-проверка после исправления frame/retrieval/policy. Этот отчёт сам по себе не доказывает готовность к включению.
