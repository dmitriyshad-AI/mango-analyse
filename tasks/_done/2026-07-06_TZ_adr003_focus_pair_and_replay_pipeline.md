> DONE 2026-07-06 07:05 | ветка codex/adr003-semanticframe-migration | codex

> TAKE 2026-07-06 06:44 | ветка codex/adr003-semanticframe-migration | codex

Ветка: codex/adr003-semanticframe-migration
Зоны: scripts/, src/mango_mvp/channels/, src/mango_mvp/integrations/, tests/, docs/, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest --collect-only -q
Семантический-аудит: да

# TZ: ADR003 focus pair + Wappi replay pipeline

Источник: `Foton/2026-07-04_TZ_REPLEY_ekzamen_100_realnyh_dialogov.md` v4, решения `D-037..D-041`, и распоряжение владельца от 2026-07-06.

## Порядок

1. Текущий ADR003-финиш приоритетнее реплея:
   - проверить итоговое состояние после `D-037..D-041`;
   - собрать фокус-пару/пакет для M1 по `reask_read`, `roles_read` и остаточному payment/#16;
   - не трогать live runtime, AMO, Tallanto, CRM и отправки.
2. Реплей-тренажер:
   - реализовать pipeline до локального pilot-10: exporter contract, pseudonymizer, slicer, runner, machine gate, judge layer/manifest;
   - полный M1 не запускать;
   - live Wappi-read не запускать без отдельного явного подтверждения владельца.

## Обязательные рамки

- Параллельность replay только по диалогам; внутри диалога ходы строго последовательно.
- Exporter read-only: без `DraftLoop.run_once`, state/journal/heartbeat, AMO/AiOffice note clients; только Wappi GET `mark_all=false`; пагинация покрыта тестом.
- Псевдонимизация покрывает сообщения, contact/chat names, manager_reference, context_used, traces и judge payload.
- `chat_only_replay` — честное имя метрики; `external_context`, `manager_issue_private`, `multi_client_burst` выделяются отдельно.
- Machine gate первичен; judge вторичен.
- Все клиентские/менеджерские тексты требуют semantic review.
