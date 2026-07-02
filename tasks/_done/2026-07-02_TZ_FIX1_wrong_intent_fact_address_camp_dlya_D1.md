> DONE 2026-07-02 20:17 | ветка codex/adr003-semanticframe-migration | codex

> TAKE 2026-07-02 19:31 | ветка codex/adr003-semanticframe-migration | codex

Ветка: codex/adr003-semanticframe-migration
Зоны: src/mango_mvp/channels/dialogue_contract_pipeline.py, tests/, audits/, tasks/, docs/worktrees_registry.md
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_subscription_llm_draft_provider.py tests/test_fact_venue_scope.py tests/test_adr003_regex_understanding_moratorium.py
Семантический-аудит: да

# ТЗ: Fix1 wrong_intent_fact для адреса и лагерного/ЛВШ контекста

## Основание

Регрейд Claude #1 диагностики 20 частых вопросов подтвердил главный кодовый дефект: `wrong_intent_fact` в `src/mango_mvp/channels/dialogue_contract_pipeline.py` ошибочно понижает корректные адресные и лагерные/ЛВШ ответы в `draft_for_manager`.

Примеры из `audits/_inbox/20260702_chastye_voprosy_live_direct_path_diag/`:

- `sm_f_address`: клиент прямо спрашивает адрес, ответ содержит корректный адрес, но gate пишет `wrong_intent_fact`: "Адресный факт нельзя выдавать как ответ на неадресный вопрос".
- `sm_u_camp1`: клиент явно говорит про ЛВШ/лагерь, но gate повторно пишет `wrong_intent_fact`: "Лагерный/ЛВШ факт нельзя выдавать как справку вне лагерного контекста".

Это баг проверки соответствия факта выходному ответу, а не новый слой понимания клиента.

## Границы

- Делать только Fix1.
- Не запускать новые ADR shadow-фазы.
- Не включать флаг в профиль.
- Не трогать живого бота `pid 60227`, Wappi, AMO, Tallanto, CRM.
- Не менять P0-floor/preblock.
- Не расширять regex-лазанью понимания клиента.
- Если меняется regex/keyword snapshot, синхронно обновить/объяснить moratorium snapshot. Лучше использовать уже имеющиеся поля `contract`, `facts`, `context`, route/fact metadata.

## Реализация

1. Добавить новый default-OFF флаг для фикса relevance gate, например `TELEGRAM_AUTHORITATIVE_GATE_SCOPE_RELEVANCE_FIX`.
2. В `_wrong_intent_fact_findings(...)` сузить ложные срабатывания:
   - адресный факт допустим, если вопрос/контракт/контекст уже классифицирован как адресный;
   - camp/ЛВШ/лагерный факт допустим, если вопрос/контракт/контекст явно относится к camp/ЛВШ/летней школе.
3. OFF должен быть строгим no-op: существующие тесты старого поведения сохраняются при выключенном флаге.
4. ON должен убрать ложный `wrong_intent_fact` только в подтверждённых областях, не снимая другие findings (`derived_product_claim`, missing facts, live availability, P0, brand, money).

## Приёмка

1. Целевые unit-тесты:
   - OFF: старое поведение для адреса/camp сохраняет `wrong_intent_fact`.
   - ON: адресный вопрос + адресный факт не даёт `wrong_intent_fact`.
   - ON: явный camp/ЛВШ контекст + camp/ЛВШ факт не даёт `wrong_intent_fact`.
   - ON: неадресный вопрос + адресный факт всё ещё ловится.
   - ON: нелагерный контекст + camp/ЛВШ факт всё ещё ловится.
2. Мораторий ADR003 зелёный.
3. Локальный замер на тех же 20 частых вопросах:
   - сколько `draft_for_manager` стало автономным или сколько `wrong_intent_fact` ушло;
   - P0/бренд/выдумка не хуже;
   - текстовые дефекты, не относящиеся к Fix1, не маскировать.
4. Отчёт в `audits/_inbox/` для Claude #1:
   - что изменено;
   - тесты;
   - baseline -> ON на 20 вопросах;
   - подтверждение, что флаг default-OFF и не включён в профиль;
   - semantic review по клиентским черновикам из 20 вопросов.

## Стоп-условия

- Нужно менять общий direct-path routing/P0/brand logic.
- Появляется новый regex/keyword-пониматель клиента вместо сужения output relevance gate.
- Локальный замер показывает brand/P0/fabrication регресс.
- Для результата требуется включить флаг в профиль или перезапуск live.
