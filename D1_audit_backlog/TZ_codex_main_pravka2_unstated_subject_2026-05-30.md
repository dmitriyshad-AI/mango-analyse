# ТЗ Кодексу main — правка 2: unstated_subject не гасит предмет из факта. 2026-05-30.

Автор: Claude #1. Волна 1 закрыта (`b29cfe2c`, 427 зелёных, проверено в песочнике). Это следующий
рычаг автономии — гвард `unstated_subject`, один из источников лишних уходов к менеджеру.

## Корень (проверено по коду b29cfe2c)

`apply_unstated_subject_guard` (`subscription_llm.py:3993`): `unexpected = _mentioned_subjects(draft) - allowed`;
если непусто → `draft_for_manager` + safe_text.

`_allowed_subjects_from_context` (4049) формирует `allowed` ТОЛЬКО из: предметов сообщения клиента
(`_mentioned_subjects(client_message)`) + `context.subject` + подтверждённых/CRM-слотов
(`dialogue_memory_view.client_confirmed_slots / crm_known_slots`). **retrieved_facts НЕ учитывает.**

Следствие: если бот упомянул предмет, реально присутствующий в ИЗВЛЕЧЁННОМ факте активного бренда
(законно — факт его содержит), но клиент его буквально не называл → предмет считается «лишним» →
ложный уход к менеджеру.

## Фикс (в `_allowed_subjects_from_context`)

Добавить в `allowed` предметы, присутствующие в retrieved_facts активного бренда
(`dialogue_contract_pipeline.retrieved_facts`), распознавая их теми же `SUBJECT_GUARD_MARKERS`.
Тогда предмет, реально извлечённый в факт хода, не считается выдумкой.

**Правило #1 — подтвердить чтением перед правкой:** доходят ли retrieved_facts до
`apply_unstated_subject_guard` через `context`. В волне 1 для operational-гварда это решалось
`_context_with_dialogue_contract_retrieved_facts(context, result)` — вероятно тот же приём нужен здесь
(передать retrieved_facts из `result.metadata` в context перед вызовом guard'а). Если retrieved_facts
до гварда не доходят — это часть правки.

## Обязательный негативный контроль (правило #4)

- предмет, которого НЕТ ни в сообщении клиента, ни в слотах, ни в retrieved_facts → по-прежнему
  «лишний» → `draft_for_manager` (реальная выдумка предмета не проходит);
- предмет из факта ДРУГОГО бренда не делает его allowed (разделение брендов не ослаблять);
- существующие тесты `unstated_subject` остаются зелёными.

## Тесты входа/выхода

- Вход: бот упомянул предмет из retrieved_facts активного бренда, клиент не называл → ложно ушёл к менеджеру.
- Выход: тот же кейс → автоответ (предмет из факта в allowed); НО предмет-выдумка (не в факте, не от
  клиента, не в слотах) → по-прежнему к менеджеру; предмет из факта чужого бренда → к менеджеру.

## Что НЕ в этой правке (вынесено)

- `_can_autonomously_replace_failed_draft` (расширение спасения автоответа) — **отдельная правка 3.**
  Причина: она ослабляет защитную реакцию `_hard_check` (пропускает забракованный verify_output
  черновик). При ответе клиенту напрямую это рискованнее, чем гвард unstated_subject — делать
  отдельно, осторожно, лучше после замера эффекта волн 1+2. Спасать можно будет только коды вроде
  `unsupported_entity` при покрытии retrieved_facts; `brand_leak / forbidden_scope / p0_promise /
  p0_semantic_risk / meta_leak / ai_disclosure` остаются жёстко fail-safe.
- Router/умолчание, `safe_template_dispatcher` — позже по дорожной карте.

## Ограничения

- Тесты не гонять (лимит). Claude #1 прогонит щит + smoke; в main не мержить до зелёного.
- Хирургично: правка в `_allowed_subjects_from_context` (+ при необходимости передача retrieved_facts
  к гварду). Не трогать subset-якоря, покрытие, P0, бренд, роутер, тёплый слой.
- Отчёт: файлы/строки, как retrieved_facts доходят до гварда, тексты позитивного и негативных тестов.
