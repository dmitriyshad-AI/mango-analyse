# Скелет тест-харнесса для длинных диалогов (15-25 ходов)

Автор: второй Claude. Дата: 2026-05-29. Read-only СКЕЛЕТ (реализует Кодекс в Phase 7-strict). Зачем: eval ~5 ходов; на 15-25 ходах ломаются латч/слоты/окна (см. `long_dialog_context_retention_analysis_2026-05-29.md`), а теста нет — без него Phase 7-strict не проверяется.

## 1. Структура и принципы

- **Фреймворк:** pytest, чистая синтетика, БЕЗ реальных LLM-вызовов. Это детерминированный тест памяти/латча/слотов/маршрута, а не качества текста.
- **Что мокаем (раннеры из `run_dialogue_contract_pipeline`, `subscription_llm.py:877-889`):**
  - `understand_fn(prompt)` → канонический dict-контракт (answerability/subquestions[].answerable/is_p0/current_question) по сценарию хода (мок «Понимания»).
  - `draft_fn(prompt)` → детерминированный текст, собранный из переданных фактов (эхо rfk) — чтобы verifier проходил.
  - `repair_fn`, `faithfulness_fn` → pass-through/«ок».
  - `warmth_fn` → identity (без изменений).
  - Реальными оставляем: `build_dialogue_memory`, `update_dialogue_memory_after_answer`, `_next_p0_latch`, `classify_answer_safety`, `tag_message_roles`, слот-извлечение, `held_state` — именно их и тестируем.
- **Драйвер диалога (ядро харнесса):**
  ```python
  def run_dialogue(scenario) -> list[TurnResult]:
      memory = None  # DialogueMemory | None
      results = []
      for turn in scenario.turns:                  # client реплики
          mem = build_dialogue_memory(
              current_message=turn.client,
              active_brand=scenario.brand,
              recent_messages=_recent(memory),      # окно как в проде
              previous_memory=memory.to_json_dict() if memory else None,  # ROUND-TRIP полной памяти
              session_id=scenario.session_id,
          )
          decision = classify_answer_safety(
              client_message=turn.client,
              context={"conversation_intent_plan": plan_view(mem),
                       "dialogue_memory_view": mem.to_prompt_view(),
                       "recent_messages": _recent_msgs(mem)},
          )
          # (опц.) прогнать v2-пайплайн с мок-раннерами для route/text
          memory = update_dialogue_memory_after_answer(mem, answer_text=mock_bot_answer, route=route, safety_flags=decision.risk_codes)
          results.append(TurnResult(turn.idx, mem, decision, route))
      return results
  ```
  Здесь же проверяется **ОТКРЫТЫЙ вопрос round-trip**: сериализуем `to_json_dict` (полную). Если прод использует `to_prompt_view` — добавить вариант теста с урезанной памятью и сравнить деградацию.
- **Генерация синтетики:** сценарий = список `Turn(idx, client, expect)`. `expect` — словарь ожиданий на этом ходу (latch.active, known_slots.grade, route_class, do_not_reask). Длинные диалоги собираются параметрически (база 6-8 ходов + вставки противоречий/эмоций/смены темы до 15-25).

## 2. Эталонные сценарии (15-25 ходов)

| # | Имя | Суть | Длина |
|---|---|---|---|
| S1 | Коррекция класса | t1 «9 класс» … t12 «нет, я про 10» (без «класс») … t13-20 цена/расписание | 20 |
| S2 | Смена темы | регулярный курс → t8 лагерь ЛВШ → t14 назад к курсу; topic-anchor/scope | 18 |
| S3 | Ложный P0 + остывание | t3 эмоция-к-боту/benign-refund → t4-20 нейтральные; латч должен ОТПУСТИТЬ | 20 |
| S4 | Реальный P0 липнет | t5 «верните деньги/в суд» → t6-20; латч ДЕРЖИТ до менеджера | 16 |
| S5 | Мульти-интент компаунд | каждый ход 2-3 подвопроса (формат+цена+скидка); coverage subqs | 15 |
| S6 | Накопление слотов | grade→subject→format→product по одному за ход; ничего не теряется к t18 | 18 |
| S7 | Бренд держится | 22 хода в одном бренде; 0 упоминаний второго; client_confirmed/scope | 22 |
| S8 | Окно-асимметрия | P0-маркер на t2, затем 10 нейтральных; P0 виден через латч, не теряется окном `[-3:]` | 15 |
| S9 | Recall из summary | договорённость на t3, ссылка на неё на t17; нужна семантическая память (7.3) | 20 |
| S10 | Противоречие формата | t2 «онлайн» → t9 «нет, очно» → t10-20; format перезаписан, без залипания | 20 |

## 3. Ожидаемые проверки по сценариям

- **S1 (коррекция класса):** на t12+ `known_slots["grade"]=="10"`; бот подтвердил смену; НЕ остался "9". Контроль: до Phase 7.2 тест КРАСНЫЙ (фиксирует баг), после — зелёный.
- **S3 (ложный P0):** `memory.p0_latch.active` True на t3, и False (RELEASED) к t5-6; route на t6+ не `manager_only`. (Phase 7.1.)
- **S4 (реальный P0):** `p0_latch.active` True с t5 до конца; route `manager_only` все ходы; авто-релиз НЕ срабатывает. **КОНТРОЛЬ безопасности — обязателен.**
- **S5 (компаунд):** `len(contract.subquestions)>=2`; оба подвопроса адресованы или честный микро-handoff; нет тихого drop.
- **S6/S7 (слоты/бренд):** к последнему ходу все названные слоты в `known_slots`+`client_confirmed_slots`; `do_not_reask` корректен; 0 упоминаний второго бренда в любом ответе.
- **S8 (окно):** claim across-turns (t2) учитывается на t13 (через латч/summary), не теряется `recent[-3:]`.
- **S9 (summary):** на t17 ответ ссылается на договорённость t3 (после 7.3 — через `conversation_summary_semantic`); до 7.3 — КРАСНЫЙ (фиксирует пробел).
- **S10 (формат):** `known_slots["format"]=="очно"` с t9; нет залипшего "онлайн".

Общие инварианты на КАЖДОМ ходу всех сценариев:
- **INV-P0:** реальный P0 (S4) никогда не теряется; ложный (S3) отпускается.
- **INV-brand:** ни один ответ не содержит обоих брендов.
- **INV-no-fab:** числа/факты в ответе только из переданных моков rfk.
- **INV-slot-monotone-correctness:** слот меняется ТОЛЬКО при явной коррекции или новом значении; иначе держится.
- **INV-prompt-size:** размер собранного drafter-промпта ≤ cap (после 7.3).

## 4. Мок-контракты (примеры фикстур)

```python
def mock_understand(turn):           # канон AnswerContract
    return {"answerability": turn.answerable, "is_p0": turn.is_p0,
            "current_question": turn.client,
            "subquestions": [{"text": q, "answerable": "self"} for q in turn.subqs]}
def mock_draft(prompt, facts):       # эхо фактов → verifier проходит
    return " ".join(facts.values()) or "Передам менеджеру уточнить."
def mock_faithfulness(_): return {"verdict": "ok"}
```
Фикстуры фактов — маленький фиксированный KB-срез (цены/скидки/refund_presale) по бренду, не реальный bot pack (синтетика, do-not-tune).

## 5. Где разместить и как запускать

- `tests/test_long_dialog_context.py` (рядом с `test_dialogue_contract_pipeline.py`, `test_fact_scope_spec.py`).
- Параметризовать `@pytest.mark.parametrize("scenario", SCENARIOS)`; каждый сценарий — отдельный кейс + per-turn assert.
- Прогон в CI отдельно от eval (eval — do-not-tune). Это фикстурный набор, не из eval.

## 6. Что харнесс НЕ делает (границы)

- Не оценивает качество/тон текста (это X2/судья, отдельно).
- Не вызывает реальные LLM (детерминизм).
- Не подменяет eval-прогон 212 (это поведение на коротких; здесь — длинные).

## Открытые вопросы

1. Round-trip памяти в проде: `to_json_dict` или `to_prompt_view`? Харнесс должен тестировать ТОТ вариант, что в проде (иначе тест зелёный, прод красный). Подтвердить у Кодекса (рантайм telegram).
2. Прогонять ли в харнессе полный v2-guard-chain (`_apply_dialogue_contract_v2_guard_chain`) или только pipeline+memory? Для проверки route/sanitize нужен весь chain; для латча/слотов — достаточно memory+classify. Предлагаю два уровня: unit (memory/latch/slots) и integration (полный v2 chain с мок-раннерами).
3. Нужны ли сценарии 25+ ходов (за `MAX_TURNS=20`)? Да — отдельный S11 «21+ ходов» для проверки front-drop turn-1 и переноса слотов через `known_slots`.
