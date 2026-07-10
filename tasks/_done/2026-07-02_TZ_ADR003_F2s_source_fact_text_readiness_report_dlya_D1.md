Ветка: codex/adr003-semanticframe-migration
Зоны: scripts/report_adr003_frame_calibration_queue.py, tests/test_report_adr003_frame_calibration_queue.py, tasks/_done/2026-07-02_TZ_ADR003_F2s_source_fact_text_readiness_report_dlya_D1.md
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_report_adr003_frame_calibration_queue.py tests/test_report_adr003_semantic_frame_eval.py tests/test_adr003_regex_understanding_moratorium.py
Семантический-аудит: да

# ADR-003 F2s: source-fact text readiness report

## Контекст

Свежий регрейд Ф2 показал, что route-only автономность почти не даёт рычага:
too-cautious в основном упирается в вопросы вида "существует ли курс/формат для класса X", а не в harmless ack/status.
Для такого ответа нужен не новый route-гейт, а подтверждённый клиентский факт и отдельная политика сборки текста.

## Решение

Добавить только отчётный слой в `report_adr003_frame_calibration_queue.py`:

- для строк `semantic_frame_proof_reconciliation_shadow.status == would_reconcile_to_safe_reference`
  открыть переданный `kb_snapshot`;
- найти сырой факт по `source_fact_key` или первому `exact_fact_keys`;
- проверить бренд, `allowed_for_client_answer`, `forbidden_for_client`, `internal_only`, `valid_until`,
  сырой `client_safe_text`, `bot_template_required` и PII-сигнал;
- вывести в отчёт только статус, длину и SHA-256 хэш `client_safe_text`, без полного текста;
- оставить `active_behavior_allowed=false` и `active_readiness=no_go`.

## Границы

- Не менять `provider.py`, `direct_path.py`, маршрут, текст, профиль, live-бота.
- Не добавлять модельные вызовы.
- Не использовать fallback `fact_text`/`manager_display_text` как клиентский текст.
- Не добавлять новые regex для понимания смысла клиента.
- PII-regex разрешён только как санитарный фильтр KB-текста в отчёте.

## Локальный замер

На свежем M1-прогоне `36ea110` proof-reconciliation trace отсутствует, поэтому readiness пустой: `0` строк.

На локальном F2q retry audit pack:

- `proof_reconciliation_would_reconcile=17`;
- `source_fact_client_safe_text_present=17`;
- `source_fact_lookup_by_status`: `found=11`, `wrong_brand=6`;
- `text_candidate_readiness_by_status`: `source_text_ready=8`, `blocked_wrong_brand=6`, `blocked_bot_template_required=3`;
- `source_fact_client_safe_text_pii_signal=0`;
- `send_as_is_review_candidates=0`;
- `active_readiness=no_go`.

Вывод: фактическая основа для части случаев есть, но активировать поведение нельзя без отдельной политики текста/шаблонов,
semantic verifier и решения по wrong-brand/template-required.
