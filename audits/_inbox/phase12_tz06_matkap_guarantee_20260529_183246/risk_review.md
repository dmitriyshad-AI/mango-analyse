# Risk Review

## Основные риски

- Смешать P0/юридическое обещание и P1-справку по маткапиталу.
- Подтвердить региональный маткапитал.
- Обещать решение СФР.
- Сломать precedence с уже мигрированными cross-brand / terminal / guarantee шаблонами.

## Как снижено

- Приоритет `40`: после result/admission guarantee, до olympiad.
- Используется существующий legacy-селектор без изменения текстов.
- `_is_verified_safe_numeric_template` уже содержит `MATKAP_FEDERAL_TIMING_SAFE_TEXT`, поэтому сроки не блокируются как unsupported numeric promise.
- Регрессии закрывают как positive, так и negative сценарии.

## Не проверялось

- Живой симулятор и LLM-поведение не запускались по ограничению этапа.
