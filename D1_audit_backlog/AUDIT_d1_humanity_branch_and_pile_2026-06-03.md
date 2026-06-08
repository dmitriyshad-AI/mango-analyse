# Аудит: ветка d1-humanity-layer + несохранённая куча главного дерева. 2026-06-03.

Автор: Клод #1 (расследование субагентом + моя верификация по git). Read-only. Цель — ответить Дмитрию: что это за
ветка, что в ней создано и зачем; и что НЕЛЬЗЯ потерять при уборке.

## 1. Что такое d1-humanity-layer и зачем (ПРОВЕРЕНО по docs ветки)
Две параллельные подзадачи одной ветки:
- **Слой «человечности»** — детерминированные гварды на ВЫХОДЕ бота (после безопасности/фактов), чтобы поднять
  «отправил бы черновик без правок» (21%→≥40%) не ослабляя безопасность: анти-повтор (`is_near_repeat`), ответ на
  новый вопрос (`unanswered_direct_question`), меньше лишних уходов при наличии факта (`should_answer_not_handoff`),
  без мета-утечек (`has_meta_leak`). Источник: `TZ_humanity_layer_2026-05-25.md`.
- **Ночная воронка (night funnel)** — SHADOW-наблюдение ночного/тестового трафика БЕЗ отправки клиенту: единственный
  владелец Telegram-апдейта пишет append-only `inbound_tee.jsonl`, отдельный replay гоняет «ночной GATE» по копии,
  без токена/reply/AMO. Обвязка под будущую вечернюю автоотправку. Источник: `TEE_SCHEME_night_funnel_shadow_2026-05-28.md`,
  `docs/MANGO_NIGHT_ACCELERATION_PLAN_2026-05-11.md`.

## 2. Коммиченная история — ВСЯ внутри принятого типа 9cc70d2b
ПРОВЕРЕНО: `git rev-list --count 9cc70d2b..36e23cb8 = 0`; 14 характерных коммитов (Wave 3a/3b/3c/7 humanity, night
funnel prep/shadow/tee, semantic state/roles, quality pipeline каркас) — каждый предок 9cc70d2b. **Терять в коммитах
нечего.** Тип ветки `codex/m1-intermediate-20260529-final` = ровно `9cc70d2b` (проверено).

## 3. Несохранённая куча — по корзинам (ПРОВЕРЕНО пофайлово)
### (a) Level A estimate-mode — НОВОЕ от Кодекса, нет в 9cc
`dialogue_contract_pipeline.py` (новые функции `_resolve_answer_mode`/`_is_product_question`/`build_estimate_prompt`/
гейт по числам), `subscription_llm.py` (коды `unsupported_product_claim`/`estimate_*`), `test_dialogue_contract_pipeline.py`
(регрессы). ПРОВЕРЕНО: символы estimate_domain/estimate_confidence/unsupported_product_claim в 9cc отсутствуют. → это и
надо перенести на 9cc (задача #103).

### (b) Дубль/отставание — при откате теряется НИЧЕГО (9cc лучше/то же)
ПРОВЕРЕНО хэшами: `dialogue_memory.py` диск == 9cc бит-в-бит (644c2ffa…). Большинство tracked-src идентичны 9cc —
статус «MM/M» лишь следствие базы на 98 коммитов позади. `rules_engine.py` на диске ПОЗАДИ (в 9cc есть
`_warm_enrollment_process_text`, на диске 0). `build_kb_release_v3_from_claude_handoff.py` диск СТАРЕЕ и даже РЕГРЕССИРУЕТ
(на диске УНПК-онлайн `manager_handoff_only`, в 9cc уже `bot_answer_self_for_pilot`). Откат tracked-правок безопасен.

### (c) ПО-НАСТОЯЩЕМУ УНИКАЛЬНОЕ (untracked, нет в 9cc) — НЕ ТЕРЯТЬ
ПРОВЕРЕНО `git cat-file -e 9cc70d2b:<path>` (отсутствуют) + `git ls-tree` (нет):
1. **WhatsApp-стек:** `scripts/whatsapp_*.py` (12 шт, вкл. `whatsapp_context_provider.py`) + `all_whatsapp_chats.txt`.
   Пайплайн обработки WhatsApp-выгрузки как нового источника контекста (read-only, P0-маркеры).
2. **KB schedule v6.4 деривация:** `scripts/derive_kb_schedule_2026_27_sources.py`, `verify_kb_schedule_2026_27_release.py`,
   `tests/test_kb_schedule_derivation.py`, и пакет `kb_release_20260602_v6_4_schedule/` (gitignored). **ОПЕРАЦИОННО НЕСУЩЕЕ
   (см. §4).**
3. **Тест-сеты:** `real_questions_20260531.jsonl` (202 реальных вопроса), `acceptance_set_v65/` (7 батчей), `scenarios_half1.jsonl`.
4. **Панель бота:** `bot_control_panel.html` + `bot_control.sample.json` (вероятно стоп-кран ночного трафика).
5. **Прогоны/бандлы:** `runs/` (3 прогона), `_bundles/bundle_agreement_3a_20260602/` (overlay для M1). Воспроизводимы.
6. Прочее: `.claude/skills/`; `Финансовая модель/` — ОТДЕЛЬНЫЙ проект пользователя, случайно в каталоге, к боту не относится.

## 4. КРИТИЧНАЯ находка (моя верификация): принятый бот зависит от НЕзакоммиченного снимка KB
ПРОВЕРЕНО: на `9cc70d2b` `DEFAULT_KB_SNAPSHOT_PATH = kb_release_20260602_v6_4_schedule/kb_release_v3_snapshot.json`, но
этот пакет в git НЕ закоммичен (нет в 9cc; на диске .gitignore его ИГНОРИТ; лежит только в куче, 10.7 МБ). При этом v6.5
(`kb_release_20260603_v6_5_summer_format_cleanup`) — закоммичен в 9cc. То есть: **код на типе указывает на снимок v6.4_schedule,
которого нет в репозитории.** Следствия:
- чистый checkout 9cc70d2b (в т.ч. `.phase12`) без локального пакета v6.4_schedule → бот/тесты не найдут снимок;
- значит §3(c).2 (schedule-деривация + пакет) — не «приятно сохранить», а ТРЕБУЕТСЯ принятой линии;
- нужно решение: либо закоммитить снимок v6.4_schedule в тип, либо перенаправить `DEFAULT_KB_SNAPSHOT_PATH` на закоммиченный
  v6.5 (надо сверить, что v6.5-снимок содержит расписание — отдельная проверка перед переключением).

## 5. Итог и рекомендация
- Коммиченная d1-история — 100% в типе, терять нечего.
- Tracked-правки кучи — Level A (перенести на 9cc) ИЛИ дубль/отставание (откат безопасен).
- Untracked (c) — при штатной уборке (`git restore`/`checkout` владельцем) НЕ удаляется; снести его может ТОЛЬКО `git clean`
  (проектом ЗАПРЕЩЁН). Значит при нормальной уборке (c) уцелеет автоматически.
- ПЕРЕД уборкой обязательно: (1) перенести Level A на 9cc (#103); (2) решить судьбу снимка KB v6.4_schedule (§4) — это
  единственный пункт, который может незаметно сломать принятого бота. Остальное (c) — закоммитить в тип по мере надобности
  (WhatsApp-стек, schedule-деривация, тест-сеты, панель), не терять.
- НИКАКОГО `git clean` без Дмитрия.
