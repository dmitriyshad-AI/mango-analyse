> DONE 2026-07-07 00:30 | ветка codex/adr003-semanticframe-migration | codex

> TAKE 2026-07-06 23:53 | ветка codex/adr003-semanticframe-migration | codex

Ветка: codex/adr003-semanticframe-migration
Зоны: src/mango_mvp/channels/subscription_llm_parts/text_hygiene.py, tests/, tasks/, audits/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest --collect-only -q tests/test_direct_p0_text_hygiene.py tests/test_subscription_llm_draft_provider.py tests/test_adr003_regex_understanding_moratorium.py
Семантический-аудит: да

# Repo-wrapper: ADR003 финальный довод до deploy-ready

Источник ТЗ: `/Users/dmitrijfabarisov/Claude Projects/Foton/2026-07-06_TZ_FINAL_dovesti_ADR003_do_deploya_dlya_D1.md`.

Цель: довести ADR003-ветку до deploy-ready без live-write: backup push, preflight, фикс #16-хвоста в резолвере kind `text_hygiene.py`, регрессии PaymentFix/P0/моратория, scrubbed P0-smoke и audit pack.

Границы:
- live/swap не трогать до отдельного подтверждения Дмитрия;
- M1 не запускать;
- fallback refund-текста не редактировать;
- маршрут P0 не смягчать;
- новый смысл через regex не добавлять, если можно использовать SemanticFrame; если расширяется детерминированный помощник, мораторий должен остаться зелёным.
