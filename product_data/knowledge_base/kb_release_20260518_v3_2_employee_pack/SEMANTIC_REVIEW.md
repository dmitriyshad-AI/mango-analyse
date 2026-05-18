# Semantic review: employees

Verdict: `PASS_WITH_NOTES`

## What passed

- formal_pass: `True`
- semantic_pass: `True`
- blocking_findings: `0`
- smoke50: `FOTON rows=25, UNPK rows=25, errors=0, brand_violations=0`

## Non-blocking risks

- У части цен, скидок, программ и лагерей нет явного `valid_until`; есть дата проверки `2026-05-18`.
- Пакет для сотрудников облегчает работу, но не заменяет решение РОПа/менеджера по спорным кейсам.
- Пакет для бота рассчитан на режим черновиков, а не автономную отправку клиенту.

## Required controls

- Перепроверять чувствительные к дате факты перед широким запуском.
- Не смешивать бренды.
- Все новые смысловые ошибки переводить в тест, фильтр или чек-лист.
