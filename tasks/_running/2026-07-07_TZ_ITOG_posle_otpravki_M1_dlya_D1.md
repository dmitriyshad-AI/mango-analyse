Ветка: codex/adr003-semanticframe-migration
Зоны: src/mango_mvp/channels/, scripts/, tests/, docs/, audits/, tasks/, product_data/telegram_dynamic_test_sets/adr003_kombo_factsel_veto_masker_ed59692b_20260707.jsonl, product_data/telegram_dynamic_test_sets/adr003_kombo_factsel_veto_masker_ed59692b_20260707_README.md
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest --collect-only -q tests/test_subscription_llm_draft_provider.py tests/test_dialogue_memory.py tests/test_semantic_reading.py tests/test_adr003_semantic_reading_e3_runner.py tests/test_adr003_regex_understanding_moratorium.py
Семантический-аудит: да

> TAKE 2026-07-07 13:22 | ветка codex/adr003-semanticframe-migration | codex

# Итоговое ТЗ Д1: окно «пока M1 меряет» + действия после экзамена

**От:** Fable 5, 07.07.2026. **Статус пакета:** `adr003_final_pachka_a246ece2_20260707` прошёл регрейд Fable — PASS (SHA/bundle/HEAD сверены, контракт ног верный, набор 248 персон полный, мини-дым fact_select живой: 4 applied conf 0.9-0.98). Пакет отдан на M1. HEAD пакета `a246ece2` заморожен — ветка дальше едет свободно, пакет защищён SOURCE_HEAD.

## Фаза 1 — пока M1 меряет (порядок = приоритет)

**1.1 Шаг E мега-ТЗ: уборка мёртвого монолита.** Function-level manifest → серия атомарных коммитов: `answer_quality_rewriter`, humanity-слои, `known_context_redundant_question_guard`, `rules_engine` dispatcher-ветка, монолитный хвост provider ниже раннего return, их мёртвые флаги (ANSWER_QUALITY_*, DRAFT_X2_*, PH2_*, Q_*, A_SELLING/COVERAGE/ESTIMATE/TRAVEL, STEP4_*, HUMANITY_*, ANTIREPEAT_STRICT, INTENT_STATE_REPAIR, SEMANTIC_DIAGNOSIS_GUARD — каждый через импорт-аудит). `dialogue_contract_pipeline` НЕ трогать без доказательства недостижимости. Критерий: pytest зелёный + смоук 10 диалогов байт-в-байт против a246ece2 (профиль тот же — тексты/route идентичны).

**1.2 Н4: B2+B5 из мега-ТЗ — сделать или письменно объяснить, почему нет.**
- B2: keyword-категории фактов (`DIRECT_PATH_CATEGORY_ALIASES` first-match) участвуют ТОЛЬКО при недоступном ретривере (timeout/empty payload); при живом LLM_RETRIEVE keyword-ветка не влияет на exact/adjacent.
- B5: `QUESTION_KIND_MARKERS` live_availability/price и `_keyword_signals` price/identity → tie-breaker: при валидном inline frame (conf≥0.90) kind/intent берётся из frame, keyword — только без frame.
Если осознанно отложил — 3 строки в running-ТЗ: причина + когда.

**D1 status 2026-07-07:** B2 закрывается регрессией: успешный LLM-retriever не смешивается с keyword-category fallback; keyword остаётся только fallback при недоступном/пустом retriever. B5 закрывается для active `intent_actions`: inline frame принимается только при `confidence>=0.90`; ранние `QUESTION_KIND_MARKERS`/`_keyword_signals` отложены, потому что они считаются до появления inline frame. Когда: отдельный двухпроходный memory/intent-plan дизайн после M1, если экзамен покажет красный класс на этих ранних маркерах.

**1.3 Заготовка «выключателя красных»:** маленький скрипт/чек-лист — как одним коммитом выключить любой из 7 флагов пачки + вернуть старые reading/apply списки для конкретного класса (на случай красных классов экзамена). Отрепетировать на локальной копии, НЕ коммитить выключения заранее.

**1.4 Slots-1b код** по `2026-07-07_DIZAIN_slots1b_pamyat_slotov_na_LLM.md` (default-OFF `TELEGRAM_SLOTS_GSF_KNOWN_MERGE`, вне пачки): merge-view, конфликт-правила, child_name-понижение, юниты+фикстуры. Замер — микро-парой ПОСЛЕ экзамена (единственный оставшийся локальный замер) → отдельное «да» → следующий цикл.

## Фаза 2 — после возврата M1 (порядок жёсткий)

**2.1** Отдать Fable сырьё OUT (transcripts+judge+REPORT+per-class счётчики обеих ног) — регрейд по чек-листу §8.
**2.2** По нумерованному «да» Дмитрия: красные классы выключить заготовкой 1.3 (зелёные уже в профиле — не трогать); latch v3 — если экзамен покажет остаток ложных ДЕНЕЖНЫХ латчей («отменили занятие→refund», «нет претензий→legal»), спроектировать v3: релиз refund/payment_dispute при [frame safe conf≥0.90] И [нет повторного денежного сигнала 3 хода] И [текущий ход — бытовой вопрос по frame], legal/child — никогда; дизайн согласовать с Fable до кода.
**2.3** Деплой: чек-лист `2026-07-07_CHEKLIST_freeze_i_svap_live_dlya_Dmitriya_D1.md` — freeze env (особо: STEP1, ручные READING/APPLY на живой машине), свап на пост-экзаменный SHA, смоук тестовой пары, откат наготове. Окно ≤ ~25.07.

## Границы
Live/AMO/Wappi — только шаг 2.3 руками с Дмитрием. Push можно. Новые понимающие regex запрещены. ≤2 итерации → СТОП. Пакет a246ece2 и его OUT не перетирать.
