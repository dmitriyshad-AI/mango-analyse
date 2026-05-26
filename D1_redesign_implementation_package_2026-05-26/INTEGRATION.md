# INTEGRATION: Фаза 1 → полная v2. Точная дельта. 2026-05-26

Канон логики — `modules/*.py` (13/13 офлайн). Ниже — что добавить/изменить в уже существующей Фазе 1 Кодекса
(`src/mango_mvp/channels/dialogue_contract_pipeline.py` + интеграция в `subscription_llm.py`), всё за флагом `TELEGRAM_DIALOGUE_CONTRACT_PIPELINE`.

## 1. Контракт → контракт-ПЛАН (modules/dialogue_understanding.py)
В Фазе 1 контракт плоский. Поднять до плана:
- Добавить в `AnswerContract`: `subquestions: [{text, answerable: self|manager, needed_fact_keys[], next_step}]`, `known_slots: {name: {value, source}}`, `client_state`.
- `build_understanding_prompt`: просить эти поля; слоты ТОЛЬКО с источником (`client_turn_N`/`fact:<key>`); needed_fact_keys только из каталога; значения не выдавать.
- `parse_contract`: парсить subquestions/slots/client_state; методы `all_needed_fact_keys()` и `assertable_slots()` (слоты только с source).
- P0 пре-гейт детерминированный — оставить, реюз реальных REFUND_RE/PAYMENT_DISPUTE_RE/legal/complaint вместо демо-регулярок модуля.

## 2. Черновик (modules/pipeline.build_draft_prompt)
- Подавать: client_state (регистр, не озвучивать эмоцию), под-вопросы, ТОЛЬКО `assertable_slots()` (несорсные слоты НЕ передавать → не утверждаются),
  facts из склада, missing (→ узкий честный хендофф), стилевые примеры (gold/few-shot по теме+бренду — НЕ источник фактов).

## 3. Выходные проверки (modules/output_verifier.py + modules/quality_layer.py)
- ЖЁСТКИЙ verifier: бренд-утечка, заземление чисел, forbidden_scope, meta/AI, P0. ВАЖНО при реюзе боевых проверок:
  - `_is_soft_number`: НЕ освобождать ≤11 целиком — заземлять проценты/части/месяцы (освобождать только класс-в-контексте/годы 2026-27).
  - бренд-токены foton: добавить голое «мфти» (не только «унпк мфти»).
- СЕМАНТИЧЕСКАЯ верность (`check_claim_faithfulness`) — отдельный дешёвый LLM-проход: не-числовые утверждения ∈ facts/слова клиента.
  **FAIL-CLOSED:** при сбое/мусоре проверки (`available=False`) автономный ответ НЕ отдавать → draft_for_manager (`semantic_check_unavailable`). НЕ PASS.
  Боевая версия должна вернуть СТРУКТУРНЫЕ claims (утверждение → чем подтверждено) и ЛОГировать «карточку доказательств».

## 4. Слой человечности (modules/quality_layer.py + pipeline.py [6])
- `form_check` (МЯГКО): штамп-зачин/повтор/нет шага/канцелярит → ТРИГГЕР X2-тепла (НЕ блок).
- `warmth_rewrite` (X2): меняет ФОРМУ, не содержание; выход ОБЯЗАТЕЛЬНО проходит полный hard-check (бренд+числа+семантика+P0); добавил смысл → отклонён, остаётся исходный.
  Можно переиспользовать существующий humanity X2-раннер как warmth_fn.

## 5. Оркестратор (modules/pipeline.run_pipeline) — полный поток
P0/manager_only → STOP; ретривал; черновик; жёсткий+семантический (fail-closed); ремонт ≤2 / fallback; form-check → X2-тепло (ре-верификация) → финал.
**Маршрут считает КОД после ретривала** (есть факты по отвечаемой части + нет P0 + нет missing), а не LLM-`answerability`.
Под-тумблеры `Toggles{enforce_slot_evidence, semantic_faithfulness, form_warmth}` — для изоляции компонента при отладке без пересборки.

## 6. Точки интеграции (реальное)
- understand_fn→LLM#1 (high reasoning); draft_fn→LLM#2; faithfulness_fn→дешёвый LLM-проход (структурные claims + лог); warmth_fn→X2.
- fact_store→`client_safe_facts_<brand>` (по fact_key → client_safe_text); catalog→реальные ключи активного бренда; история→НАСТОЯЩАЯ поролевая.
- style_examples→gold/few-shot по теме+бренду (углубить корпус до 3-5 вариативных на тему — иначе штамп).

## 7. Прунинг гвардов и §13.7
- По `GUARDS.md`: оставить только верификаторы безопасности; понимающе-переписывающий каскад в v2-ветке отключить; autonomy_matrix — только route-часть.
- Держать §13.7: в v2 `draft_text` меняют ТОЛЬКО safety-коррекция/фоллбэк, X2-тепло (после проверок), sanitize. См. `INVARIANT_13_7_PROOF.md`.

## 8. Тесты на сборке (добавить к моим 13)
Кодекс добавляет рантайм-тесты: реальный LLM падает/кривой JSON faithfulness (fail-closed), противоречивый контракт, missing-факт по ОДНОМУ подвопросу составного сообщения, латентность (лог).
