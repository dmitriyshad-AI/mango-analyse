# База знаний для бота

Это машинный пакет для Telegram/email/CRM-бота. Он содержит полные реестры фактов, фильтры, правила брендов и отчёты проверок.

## Статус

- run_id: `kb_release_20260602_v6_4_schedule`
- formal_pass: `True`
- semantic_pass: `True`
- blocking_findings: `0`
- smoke_status: `FOTON rows=0, UNPK rows=0, errors=0, brand_violations=0`

## Главные файлы

- `client_safe_facts_foton.jsonl` — факты, которые можно использовать для Фотона.
- `client_safe_facts_unpk.jsonl` — факты, которые можно использовать для УНПК.
- `manager_only_or_internal_facts.jsonl` — факты только для проверки менеджером/внутренней логики.
- `facts_registry.jsonl` — полный реестр с разрешениями и маршрутами.
- `post_filter_registry.json` — запретные фразы и фильтры.
- `bot_template_registry.json` — обязательные шаблоны для фактов, которые нельзя подставлять дословно.
- `bot_fact_index.json` — компактный индекс по брендам, типам и маршрутам.
- `BOT_USAGE_CONTRACT.md` — правила использования в боте.

Бот не должен отправлять сообщения клиентам напрямую на первом этапе. Он готовит черновик и показывает менеджеру.
