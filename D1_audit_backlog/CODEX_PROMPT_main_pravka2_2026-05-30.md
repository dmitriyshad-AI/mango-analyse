# Стартовый промт — Кодекс main (правка 2: unstated_subject). 2026-05-30.

> Скопировать в окно Кодекса main.

---

Ты — Кодекс main, ветка `codex/m1-intermediate-20260529-final`, HEAD `b29cfe2c` (волна 1 закрыта,
427 зелёных по прогону Claude #1).

**Прочитай:** `D1_audit_backlog/TZ_codex_main_pravka2_unstated_subject_2026-05-30.md` (корень проверен по коду).

**Суть:** `_allowed_subjects_from_context` (subscription_llm.py:4049) НЕ учитывает retrieved_facts →
предмет из реального извлечённого факта, не названный клиентом, считается «лишним» → ложный уход к
менеджеру. Добавить в allowed предметы из `dialogue_contract_pipeline.retrieved_facts` активного бренда
(по SUBJECT_GUARD_MARKERS).

**Правило #1:** подтвердить чтением, доходят ли retrieved_facts до `apply_unstated_subject_guard`; если
нет — дотащить так же, как в волне 1 для operational (`_context_with_dialogue_contract_retrieved_facts`).

**Негативный контроль обязателен:** предмет НЕ из факта/слотов/сообщения → по-прежнему к менеджеру;
предмет из факта ЧУЖОГО бренда → к менеджеру; существующие unstated-тесты зелёные.

**НЕ в этой правке:** `_can_autonomously_replace` (это правка 3), router, safe_template, покрытие, P0, бренд.

**Ограничения:** тесты не гонять (лимит) — Claude #1 прогонит щит + smoke; в main не мержить до
зелёного; хирургично. Отчёт: файлы/строки, как доходят retrieved_facts, тексты позитивного и
негативных тестов.
