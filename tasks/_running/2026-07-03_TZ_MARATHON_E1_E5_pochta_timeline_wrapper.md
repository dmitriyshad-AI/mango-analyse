> TAKE 2026-07-02 03:21 | ветка codex/email-pipeline-restore | codex

Ветка: codex/email-pipeline-restore
Зоны: src/mango_mvp/customer_timeline/, src/mango_mvp/channels/, src/mango_mvp/existing_clients/, src/mango_mvp/amocrm_runtime/, scripts/, tests/, docs/, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_customer_timeline_a2_mail_ingest.py tests/test_customer_timeline_store.py tests/test_customer_timeline_derived_signals.py tests/test_email_pipeline_brand.py
Семантический-аудит: да

# Repo-обвязка для марафона Э1-Э5: почта + Customer Timeline

Это техническая обвязка для штатного `scripts/preflight.py`. Источник задач:

- `/Users/dmitrijfabarisov/Claude Projects/Foton/2026-07-03_TZ_MARATHON_E1_E5_pochta_timeline_dlya_Codex2.md`
- `/Users/dmitrijfabarisov/Claude Projects/Foton/2026-07-02_TZ_E1_v2_pochta_fakty_botsafe_enrich_dlya_Codex2.md`
- `/Users/dmitrijfabarisov/Claude Projects/Foton/2026-07-02_PLAN_v2_treka_pochta_timeline_i_TZ_E4a_podklyuchenie.md`

Журнал решений Codex 2:

- `/Users/dmitrijfabarisov/Claude Projects/Foton/2026-07-03_MARATHON_DECISIONS_Codex2.md`

Жёсткие локальные уточнения из Decision Pack 0:

- все записи только в staging под `.codex_local/staging/` и локальные рабочие артефакты;
- prod-БД только `mode=ro&immutable=1`, без writable-open/checkpoint;
- CRM/AMO/Tallanto без write, только read-method allowlist и export package;
- hard safety issue после аудита этапа = STOP, не known limitation;
- `mail_archive_stage2` может открываться боту только staging-only и за флагом default OFF;
- при конфликте старого плана 2026-07-02 и марафона 2026-07-03 приоритет имеет марафон.

ПДн, токены, содержимое писем, телефоны и email клиентов не писать в git/Foton.
