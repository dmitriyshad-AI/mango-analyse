Ветка: codex/adr003-semanticframe-migration
Зоны: src/mango_mvp/channels/subscription_llm_parts/policy_routing.py, tests/, docs/ADR003_REGEX_UNDERSTANDING_MORATORIUM.md, tests/test_adr003_regex_understanding_moratorium.py, audits/_inbox/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest --collect-only -q
Семантический-аудит: да

# Wrapper PR-A Fix1b

Исполнять только PR-A из внешнего ТЗ:
`/Users/dmitrijfabarisov/Claude Projects/Foton/2026-07-03_TZ_FIX_zahod_fix1b_slots_reask_dlya_D1.md`.

Границы:

- не трогать PR-B/slots-merge, legacy grade/subject/format regex, known_slots;
- якориться по reason-кодам `autonomy_default_cautious_missing_facts` и `autonomy_default_cautious_unverified_fact`, не по номерам строк;
- live-status demote и output-floor availability/promise не обходить;
- corridor должен проверять весь черновик по свежим client-safe фактам и не ослаблять P0/high-risk;
- три partial-support стоп-юнита обязательны: лишнее число, чужой бренд, живые места;
- флаг `TELEGRAM_FIX1B_AUTONOMY_VERIFIED_FACTS` default-OFF, не в профиль.
