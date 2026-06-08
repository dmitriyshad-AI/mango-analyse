# Разбор трассировки: механика лишних уходов к менеджеру. 2026-05-30.

Автор: Claude #1. Источник: `runs/20260529_014208_eval_FINAL_part_b_TRACE/debug_trace.jsonl`
(564 записи, 16 диалогов Части B), сверено с кодом payload HEAD `6f849cff`. Все утверждения о коде
проверены по сырым строкам (номера ниже), не по пересказу.

## Краткий вывод

Лишний уход к менеджеру (over_handoff) — не баг одного места. Решение «уйти к менеджеру»
принимается в ТРЁХ независимых слоях с разными правилами, и поздний слой перебивает ранний.
Понимание (модель) в ряде ходов говорит «бот может ответить, факты есть», но детерминированные слои
всё равно уводят. Это структурный корень, а не настройка.

## Карта узлов (что прогон делает на каждом ходе)

| Узел | Частота | Роль |
|---|---|---|
| `_hard_check` | 120 (≈2/ход) | жёсткая проверка готового черновика |
| `p0_pre_gate` | 69 | предварительная P0-проверка |
| `understand` | 60 | понимание: даёт `answerability` (answer_self / manager_only) |
| `apply_unsupported_promise_guard` | 60 | гвард обещаний (меняет route_before→route_after) |
| `safe_template_dispatcher` | 60 | подмена на верифицированный шаблон |
| `_apply_dialogue_contract_v2_guard_chain` | 60 | цепь гвардов + route_permission |
| `build_draft` | 56 | черновик (модель пишет прозу) |
| `retrieve_facts` | 51 | извлечение фактов |
| `coverage_check` | 16 | проверка покрытия |
| `_safe_fallback_text` | 12 | текст ухода к менеджеру |

Поток на ход: `p0_pre_gate → understand → retrieve_facts → build_draft → _hard_check(×) →
[coverage] → apply_unsupported_promise_guard → safe_template_dispatcher → guard_chain(+route_permission)`.

## Что показал прогон (эмпирика)

- `understand.answerability`: **25 answer_self / 35 manager_only** из 60 ходов (B — трудная выборка:
  P0, возврат, лагерь — высокая доля manager_only ожидаема).
- Финальные маршруты guard_chain: **автоответ 26, черновик менеджеру 20, manager_only 14**. Автоотвечено
  меньше половины.
- **12 уходов через `_safe_fallback_text`.** Из них ТРИ — явно лишние (понимание сказало answer_self
  И факты были):
  - `S4_camp_who_05` t2: answer_self, 2 факта, вопрос «для какого возраста смена?» → уход.
  - `C1_format_price_04` t1: answer_self, **9 фактов**, «6 класс математика онлайн/очно и цена» → уход.
  - `E1_price_05` t1: answer_self, 3 факта, «онлайн или очно, цена для 10 класса» → уход.
  - (Ещё несколько с answerability=manager_only при доступных фактах — напр. `C3_camp_refund` t1 при
    12 фактах — но это осторожность слоя ПОНИМАНИЯ, отдельный пласт.)
- **7 изменений маршрута гвардами:** `safe_template_dispatcher` 5× понизил bot_answer_self →
  draft_for_manager (`E7_olymp` t1-t4, `S3` t3); `apply_unsupported_promise_guard` 1× → manager_only
  (`C2` t1, blocked=True); 1× промоушн (`C3` t4).

## Где именно теряется автоответ (проверено по коду)

Три слоя демоушена, поздний перебивает ранний:

1. **Ранние гейты `run_pipeline`** (`dialogue_contract_pipeline.py:898+`) — ~10 последовательных
   return-ов + флаг `force_draft_for_manager`.
2. **`_hard_check` (1468) + `_can_autonomously_replace_failed_draft` (2714).** Готовый черновик
   проверяется; при находке его можно спасти в автоответ ТОЛЬКО если ВСЕ находки = `fact_grounding`
   (числовой дрейф). Дословно (2714): `return all(finding.code == "fact_grounding" for finding in
   findings)`. Любая другая находка (бренд, мета, P0, чужая сущность, неверный scope) → уход к
   менеджеру. Плюс: если судья недоступен (`semantic_check_unavailable`) → тоже уход.
3. **Пост-цепь `_apply_dialogue_contract_v2_guard_chain` (`subscription_llm.py:1273`) +
   `_dialogue_contract_v2_route_permission_guard` (1417).** Перебивает автоответ пайплайна.
   route_permission гасит автоответ, если: forced_manager_only; high-risk маркер; бренд unknown; или
   **автономия не включена в контексте** (`result.route in AUTONOMOUS_ROUTES and not
   _autonomy_enabled(context)`, 1454) → draft_for_manager.

## Корневой множитель ложных уходов (проверено)

`_claim_supported_by_facts` (`subscription_llm.py:7740`): `return any(normalized_claim in text for
text in normalized_facts)` — фраза считается подтверждённой, только если она почти ДОСЛОВНО входит в
текст факта (плюс 2 хардкода про «до 1 июля/июня»). Живая продающая фраза («менеджер свяжется с
вами», «обычно есть и вечерние группы», упомянутый по ходу предмет) дословно в факт не попадает →
считается необоснованной → уход. На этом узле висят гварды `unsupported_promise`,
`unconfirmed_operational_specificity`, `unstated_subject`.

Скрытый множитель: `_fresh_fact_texts` обнуляет ВСЕ факты, если контекст помечен «несвежим» → тогда
любой факт «не подтверждает» → over_handoff резко растёт независимо от остального.

## Связь с метриками

over_handoff = 22% ходов в Части A, 35% в Части B. Все провальные диалоги регрейда (S4, C2, C3,
гипотетический возврат V1) помечены `over_handoff`/`ignored_question`. Когда бот не уверен, он либо
уходит к менеджеру (часто и лишне), либо реже — выдумывает (S4/C2).

## Вывод для пересборки

Понимание (модель) и факты часто ДОСТАТОЧНЫ для ответа, но три параллельных слоя демоушена +
дословная проверка покрытия гасят автоответ. Это ровно зафиксированный «принципиальный изъян»:
детерминированный слой не верифицирует точечно, а плодит дублирующие точки ухода. Лечится сведе́нием
маршрута в одну точку + семантической (не дословной) проверкой покрытия. См.
`PLAN_peresborka_sloya_marshruta_2026-05-30.md`.
