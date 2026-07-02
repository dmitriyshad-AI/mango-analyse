# Risk Review

## Runtime-риск

Низкий: изменён только offline report script. Direct-path, provider, profile,
live bot, P0-floor/preblock и route/text клиента не менялись.

## Риск неверного вывода

Основной риск - классификация `requires_fact_assertion` строится по gold notes.
Это приемлемо для отчёта/triage, но не является runtime policy.

## Guard

Отчёт явно держит `active_readiness=no_go` и не даёт разрешения включать Ф3.
`clean_route_only_discussion=0`, поэтому active-патч из этих данных делать нельзя.
Negative controls показывают 29 строк настоящих live/operational запросов,
которые должны оставаться в менеджерском контуре.
