# ТЗ-137 — ADR-002 direct: слоты + keyword fallback

Дата: 2026-06-17
Ветка: `codex/tz137-adr002-direct-slots-fallback`
Статус: `formal_pass`, нужен регрейд Claude по микро-набору перед включением флагов.

## Что сделано

1. Флаг A `TELEGRAM_DIRECT_PLAN_KNOWN_SLOTS`, default OFF:
   - direct path при ON читает `conversation_intent_plan.known_slots`, fallback на старый `slots` оставлен для replay;
   - OFF оставляет старое чтение `slots`;
   - `do_not_reask_slots` при ON строится только по подтверждённым слотам: `slot_provenance` с quote/source или `client_confirmed_slots`;
   - неподтверждённые CRM/контекстные слоты не становятся запретом на уточняющий вопрос.

2. Флаг B `TELEGRAM_DIRECT_KEYWORD_FALLBACK_RELEVANCE`, default OFF:
   - при ON keyword fallback больше не берёт `fallback_core` и `list(records)[:max_facts]` как широкий снимок;
   - fallback оставляет только факты с реальным совпадением по категории, слоту, токену вопроса или обязательному шаблонному факту;
   - `empty_selection/timeout` + открытый вопрос теперь может запускать повторную рубрику даже при пустом `facts`;
   - если итог всё ещё `draft_for_manager` без `missing_facts`, включается узкий переспрос вместо ухода к менеджеру;
   - P0/high-risk не переписываются.

3. Флаг C `TELEGRAM_DIRECT_SLOT_TOPIC_SHADOW`, default OFF:
   - добавлен теневой вызов модели для извлечения слотов/темы;
   - результат пишется только в `metadata["direct_path"]["slot_topic_shadow"]`;
   - не влияет на `known_slots`, `slot_scope`, выбор фактов, prompt, route или текст ответа;
   - fail-soft: timeout/runtime/invalid JSON логируются в metadata, ответ не меняется.

## Проверки

- `git diff --cached --check` — passed.
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m py_compile ...` — passed.
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_conversation_intent_plan.py tests/test_subscription_llm_draft_provider.py -q` — passed.
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q` — `3339 passed, 2 skipped, 1 warning in 47.80s`.

## Что не запускалось

- Микро-замер OFF→ON на `real_unpk_format_10`, история, IT/11 и NEG-кейсах не запускался. Это отдельный симуляторный регрейд для Claude.
- Флаги не добавлялись в `pilot_gold_v1` и не включались в боевой профиль.

## Остаточные риски

- A может увеличить `scope_demoted_ids`, если включить без `TELEGRAM_ASSUMED_SCOPE_GUARD`; это нужно мерить в паре, как указано в ТЗ.
- B снижает широкий fallback, но может потребовать калибровки на NEG-кейсах, где широкий пакет случайно помогал.
- C только наблюдает; решение о переносе модельных слотов в боевую логику не принималось.
