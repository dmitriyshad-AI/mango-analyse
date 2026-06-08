# Глубокий аудит `subscription_llm.py` — порядок и перекрытия guard'ов

Автор: второй Claude. Дата: 2026-05-29. Read-only. Прочитаны: оркестратор `build_draft` (`:817`), v2-цепочка (`:994-1029`), legacy-стек (`:826-866`), каскад шаблонов (`apply_subscription_policy_guards:2120`, `apply_high_risk_content_guards:2186`, фрагмент `:2320-2389`), композиции identity (`:1316/1389/1475`). Файл ~4000 строк; покрытие чтением ~15-20%, но ключевая управляющая логика прочитана. Неподтверждённое помечено «ОТКРЫТО».

## Главная находка: ДВА расходящихся пайплайна

`build_draft` (`:817`) ветвится по `dialogue_contract_pipeline_enabled(context)` (`:823`):

- **v2 (пилот/eval)** — `_build_dialogue_contract_pipeline_draft` (`:868`) → `_apply_dialogue_contract_v2_guard_chain` (`:994`). Eval-прогон идёт ЗДЕСЬ (route `bot_answer_self_for_pilot`, `:891`).
- **legacy** — `else` (`:826-866`): `build_draft_prompt` + ~18 guard-вызовов, **применённых ДВАЖДЫ** (блок `:829-838` ≈ `:848-857`) + autonomy/humanity/x2.

**Следствие №1 (safety-gap):** v2-цепочка по комментарию (`:1001`) — «safety verifiers only; **no old intent/template rewrites**». То есть весь legacy-каскад из ~30 безопасных шаблонов (`apply_subscription_policy_guards`/`apply_high_risk_content_guards`) в v2 **НЕ вызывается**. Шаблоны `_olympiad_online_safe_template`, `_camp_safe_template`, `_format_choice_safe_template`, `terminal/matkap/tax/...` — **мертвы в активном пилотном пути**. Это объясняет, почему часть hard_gate (E7 олимпиада-класс, S3 closed-world) проскочила в eval: защищающие шаблоны живут в legacy, а eval идёт через v2.

**Следствие №2 (двойное сопровождение):** guard добавленный в один путь легко забыть во втором → расхождение поведения между конфигами.

## Порядок применения — v2-цепочка (активный путь)

`_apply_dialogue_contract_v2_guard_chain` (`:1002-1029`), каждый шаг (1-6) с `_reverify_dialogue_contract_text_change` (`:1031`) после:

| # | guard | строка | может менять |
|---|---|---|---|
| 1 | apply_payment_confirmation_guard | 1003 | route/text |
| 2 | apply_brand_separation_guard | 1007 | route/text |
| 3 | apply_input_policy_guards | 1011 | route/text (+ identity? ОТКРЫТО) |
| 4 | apply_unstated_subject_guard | 1015 | text→safe, route |
| 5 | apply_unsupported_promise_guard | 1019 | route→manager_only |
| 6 | apply_unconfirmed_operational_specificity_guard | 1023 | text→safe |
| 7 | apply_funnel_policy_guard | 1027 | route |
| 8 | _dialogue_contract_v2_route_permission_guard | 1028 | **route (после контента!)** |
| 9 | _sanitize_dialogue_contract_client_text | 1029 | **text (последним!)** |

Плюс x2-warmth — ВНУТРИ `run_dialogue_contract_pipeline` (`:885`), не в этой цепочке.

**Перекрытия в v2:** (а) шаги 8-9 идут ПОСЛЕ контентных guard'ов — route_permission может перекинуть маршрут, а sanitize переписать текст уже после всех проверок (это и есть механизм утечки «Клиент спрашивает…» из `_safe_fallback_text`). (б) `_reverify` после каждого шага — хорошо для безопасности, но порядок-зависим: шаг 5 видит выход шага 4; перестановка меняет исход.

## Порядок — legacy (если конфиг переключат)

`:829-866`: payment_confirmation → brand_separation → input_policy → conversation_intent_plan → high_risk_content(**=template cascade**) → unstated_subject → unsupported_promise → unconfirmed_operational → known_context_redundant → funnel → answer_quality_rewriter → **(тот же блок ещё раз `:848-857`)** → autonomy_matrix → humanity → humanity_x2.

**Двойной проход** (`:829-838` и `:848-857`) предполагает ИДЕМПОТЕНТНОСТЬ каждого guard'а. Если guard не идемпотентен (добавляет флаг/текст при каждом вызове) — двойное применение даёт дубль/осцилляцию. Это латентный риск при добавлении нового не-идемпотентного guard'а.

## Каскад шаблонов — ручная взаимоисключаемость (хрупкая)

`apply_subscription_policy_guards` (`:2120`) и далее: каждый шаблон гейтится РАСТУЩЕЙ цепочкой `"" if cross_brand_guarded() or skip_green_template_overwrite or metadata.get("terminal_safe_template_applied") or metadata.get("direct_process_safe_template_applied") or ... else _X_safe_template(...)` (примеры `:2370 matkap`, `:2378 tax`, `:2386 camp`). Precedence = «кто применился первым, ставит `metadata["X_applied"]=True`, остальные себя пропускают».

**Риск:** добавление нового шаблона требует вручную добавить его флаг в skip-условие ВСЕХ последующих шаблонов. Забыли один → два шаблона перезапишут `draft_text` подряд (последний победит, первый «съеден»). Это комбинаторная зависимость на ~30 шаблонах при ~5% покрытия тестами.

**Перекрытие route:** шаблон делает `route = "manager_only" if route=="manager_only" else "draft_for_manager"` (`:2356/2372/2380`) — то есть шаблон может ПОНИЗИТЬ автономию и заменить весь `draft_text`, перекрывая результат раннего guard'а.

## Guard'ы, которые отменяют/перекрывают друг друга

1. **sanitize (v2 шаг 9, `:1029`)** — последним переписывает текст; может «откатить» аккуратный текст раннего guard'а или, наоборот, пропустить утечку из шаблона (наблюдали «Клиент спрашивает»).
2. **route_permission (v2 шаг 8, `:1028`)** — меняет route после того, как контентные guard'ы уже решили; может вернуть автономию там, где guard хотел manager, или наоборот.
3. **Каждый safe-template перезаписывает `draft_text` целиком** — стирает работу драфтера и предыдущих текстовых guard'ов (legacy).
4. **identity_disclosure** (`guard_identity_disclosure:1556`) вызывается в legacy-композициях (`:1316/1389/1475`), но в v2-цепочке НЕ виден явно — **ОТКРЫТО**: входит ли он в `apply_input_policy_guards`/pipeline. Если нет — identity-защита РАЗНАЯ в v2 и legacy (в eval identity_disclosure_guarded всё же появлялся — значит он где-то в v2-пути; подтвердить место).
5. **`unsupported_promise` → manager_only**, затем **funnel/route_permission** могут вернуть autonomous — порядок решает финал.

## Мёртвые ветки (в активном v2-пилоте)

- **Весь legacy-путь** `:826-866` — если `dialogue_contract_pipeline_enabled` всегда True в пилоте (ОТКРЫТО: подтвердить). Тогда ~половина guard-стека дремлет.
- **Каскад из ~30 safe-шаблонов** (`apply_subscription_policy_guards`/`apply_high_risk_content_guards`) — не вызывается в v2.
- **answer_quality_rewriter, autonomy_matrix, known_context_redundant** — в v2-цепочке отсутствуют (есть в legacy).
- `green_terminal_template`/`skip_green_template_overwrite` ветки (`:2331-2340`) — срабатывают лишь при специфичных комбинациях; часть путей может не достигаться.

Мёртвый код опасен двояко: (а) сопровождение/путаница (~2000+ строк дремлют), (б) ложное чувство защиты — «у нас есть шаблон на олимпиаду-класс», хотя в пилоте он не работает.

## Side-by-side: какие guard'ы в каком пути

| guard | v2-цепочка (пилот) | legacy (двойной проход) | примечание |
|---|---|---|---|
| payment_confirmation | да (`:1003`) | да (`:829`) | оба |
| brand_separation | да (`:1007`) | да (`:830/848`) | оба |
| input_policy_guards | да (`:1011`) | да (`:831/849`) | identity внутри? ОТКРЫТО |
| conversation_intent_plan_guard | **НЕТ** | да (`:832/850`) | расхождение |
| high_risk_content_guards (=шаблоны) | **НЕТ** | да (`:833/851`) | **safety-gap: шаблоны мертвы в v2** |
| unstated_subject | да (`:1015`) | да (`:834/852`) | оба |
| unsupported_promise | да (`:1019`) | да (`:835/853`) | оба |
| unconfirmed_operational | да (`:1023`) | да (`:836/854`) | оба |
| known_context_redundant | **НЕТ** | да (`:837/855`) | расхождение |
| funnel_policy | да (`:1027`) | да (`:838/856`) | оба |
| answer_quality_rewriter | **НЕТ** | да (`:839`) | расхождение |
| autonomy_matrix | **НЕТ** | да (`:857`) | расхождение |
| route_permission (v2) | да (`:1028`) | — | только v2 |
| humanity / humanity_x2 | x2 внутри pipeline (`:885`) | да (`:858-859`) | разные точки |
| sanitize client text | да (`:1029`) | (внутри гайрдов) | v2 — последним |
| subscription_policy_guards (шаблоны+identity) | **НЕТ** | да (`:1475`-композиция) | **мертвы в v2** |

Вывод таблицы: в v2 ОТСУТСТВУЮТ ~6 классов обработки, что есть в legacy (intent-guard, шаблоны, quality-rewriter, autonomy-matrix, known-context). Часть — намеренно (v2 = «verifiers only»), но среди отсутствующих — **безопасные шаблоны олимпиады/лагеря/closed-world**, чьё отсутствие в v2 и даёт hard_gate-проскоки. Это не «лишний код» — это незакрытая в v2 защита.

## Worked example: как утекает «Клиент спрашивает …»

Трасса v2 на компаунд-вопросе без покрытого факта (реконструкция `C1_format_price_04 t1`):
1. pipeline (`run_dialogue_contract_pipeline`) → `_hard_check` валит draft → `_safe_fallback_text` ставит текст «Передам менеджеру уточнить именно это: **Клиент спрашивает**, …» (3-е лицо в шаблоне).
2. v2-цепочка `:1015-1023` — текстовые guard'ы не трогают (не их паттерн).
3. `_sanitize_dialogue_contract_client_text` (`:1029`, ПОСЛЕДНИЙ) — НЕ ловит «Клиент …» (нет правила 3-го лица; см. Волна 2 / 11.12).
→ утечка проходит до клиента. Корень: текст рождён шаблоном-фоллбэком, а sanitize-последний его не чистит. Это и есть «перекрытие наоборот»: ни один guard не отменил, потому что паттерна нет, а источник (fallback-шаблон) — вне зоны их проверки.

## Скрытые регрессии при добавлении новых guard'ов

1. Добавить в v2, забыть legacy (или наоборот) → дивергенция.
2. Новый шаблон без обновления skip-цепочек последующих → двойной шаблон/перезапись.
3. Не-идемпотентный guard в legacy-двойном проходе → дубль флагов/текста.
4. Вставка guard'а ПОСЛЕ sanitize/route_permission vs ДО — разный исход (порядок не задекларирован, держится на последовательности кода).
5. Guard, меняющий текст, в v2 триггерит `_reverify` каждый раз → латентность + риск осцилляции, если re-verify сам инициирует правку.
6. `except Exception` в раннерах (`:954`, `:974`) глушит ошибку LLM → пустой/raw → тихий fallback на safe-template/пустой draft.

## Топ-10 самых рискованных мест для будущих волн

1. **Дивергенция v2↔legacy** — два пайплайна, нет единого источника порядка guard'ов.
2. **Мёртвый каскад шаблонов в v2** — safety-gap (защиты не применяются в пилоте) + dead code.
3. **Ручные `or metadata.get("..._applied")` skip-цепочки** (~30 шаблонов) — забыл флаг → двойной шаблон.
4. **Legacy двойной проход** (`:829-838`/`:848-857`) — допущение идемпотентности.
5. **sanitize/route_permission последними** (`:1028-1029`) — поздняя перезапись текста/маршрута; источник утечки «Клиент …».
6. **identity-guard разный в v2 и legacy** (ОТКРЫТО) — несогласованная защита природы бота.
7. **`except Exception` в LLM-раннерах** (`:954/974`) — тихий fallback, маскирует сбои.
8. **Каждый шаблон перезаписывает `draft_text` целиком** — стирает работу драфтера/верификатора.
9. **`_reverify` после каждого guard в v2** — порядок-зависимость + стоимость + потенциальная осцилляция.
10. **~344 точки guard/template/except при ~5% покрытии** — комбинаторика взаимодействий не тестируется; регресс-suite на стек отсутствует.

## Как обнаружить каждый топ-риск (детект для Кодекса)

1. **Дивергенция v2↔legacy:** diff списков guard-вызовов `:1002-1029` vs `:829-866`; CI-тест «множества guard'ов совпадают (или явно задекларировано отличие)».
2. **Мёртвый каскад в v2:** покрытие — какие `*_safe_template` НЕ достигаются при `dialogue_contract_pipeline_enabled=True`; прогнать eval-входы и собрать сработавшие шаблоны (ожидается ∅) → подтвердить мёртвость.
3. **Skip-цепочки шаблонов:** статический обход — у каждого `_X_safe_template` собрать его `metadata`-флаг и проверить, что он есть в skip-условии ВСЕХ последующих; отсутствие → потенциальный двойной шаблон.
4. **Двойной проход legacy:** тест идемпотентности (ниже).
5. **sanitize/route_permission последними:** тест «guard выставил route=manager_only / текст X → после sanitize+route_permission route/текст не изменились вопреки намерению».
6. **identity в v2:** прогнать вход с identity-фразой в драфте через v2 → флаг `identity_disclosure_guarded` должен подняться; если нет — защита отсутствует в v2.
7. **except-swallow:** инъекция исключения в раннер → проверить, что fallback логирует причину, а не молчит.
8. **Перезапись draft_text шаблоном:** лог «draft до/после каждого шаблона», diff.
9. **re-verify осцилляция:** счётчик `_reverify` на ход; >N → флаг.
10. **Комбинаторика:** регресс-suite вход→(guard, route, flags) на синтетике (тест-харнесс, Задача 3).

## Чек-лист идемпотентности (двойной проход legacy `:829-838`/`:848-857`)

Для КАЖДОГО из guard'ов проверить `g(g(x)) == g(x)` по route/text/flags/checklist:
- `apply_unsupported_promise_guard` — добавляет флаг `unsupported_promise_detected` (через `dict.fromkeys`, дедуп — вероятно идемпотентен; проверить metadata `unsupported_promises` — не дублирует ли список).
- `apply_unstated_subject_guard` — заменяет draft на safe-text (идемпотентен, если safe-text стабилен).
- шаблоны (`apply_subscription_policy_guards`) — гейтятся `metadata.get("..._applied")` → второй проход себя пропускает (идемпотентны ПО ФЛАГУ, но только если флаг проставлен на первом).
- `apply_funnel_policy_guard`, `apply_answer_quality_rewriter`, `apply_humanity_*` — НАИБОЛЕЕ подозрительны на не-идемпотентность (rewriter может переписать переписанное); приоритет проверки.

## Рекомендации (для отдельной backlog-волны, не Phase 11)

1. **Единый декларативный реестр guard'ов** с явным порядком и инвариантом приоритета (safety > brand > scope > content > template > route > sanitize), общий для v2 и legacy.
2. **Решить судьбу legacy/шаблонов:** либо перенести нужные safe-шаблоны в v2 (закрыть safety-gap по олимпиаде-классу/closed-world), либо удалить мёртвый код. Сейчас «защита есть, но не в активном пути» — худший вариант.
3. **Заменить ручные skip-цепочки** на один диспетчер «выбери максимум один шаблон по приоритету».
4. **Тест идемпотентности** каждого guard'а (применить дважды = применить один раз).
5. **Регресс-suite на стек:** таблица «вход → какой guard сработал → финальные route/text/flags», прогон на синтетике (см. тест-харнесс, Задача 3).
6. Заменить голые `except Exception` на узкие + логирование причины fallback.

## Открытые вопросы

1. `dialogue_contract_pipeline_enabled` — всегда ли True в пилоте/eval? (тело не дочитал). От этого зависит, мёртв ли legacy полностью.
2. Входит ли `guard_identity_disclosure` в v2-путь (через `apply_input_policy_guards` или pipeline)? В eval флаг появлялся — значит где-то да; подтвердить точку.
3. Идемпотентны ли все guard'ы legacy-двойного прохода? Нужен прогон-проверка.
4. `_dialogue_contract_v2_route_permission_guard` (`:1028`) — может ли вернуть автономию после `manager_only` от unsupported_promise? (тело не читал).
5. Полный список шаблонов в каскаде и их взаимные skip-условия — нужен механический обход (не весь файл прочитан).
