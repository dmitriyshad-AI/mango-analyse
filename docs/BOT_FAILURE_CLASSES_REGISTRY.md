# Bot Failure Classes Registry

Дата создания: 2026-05-23

Назначение: единый реестр классов проблем Telegram-ботов Фотона и УНПК МФТИ. Этот файл нужен, чтобы не чинить каждый FAIL точечно, а закрывать повторяемые причины.

## Правило работы

Каждый подтвержденный FAIL/PASS_WITH_NOTES из динамических тестов, статичных тестов, пилота, feedback сотрудников или ревью Claude/Codex должен быть отнесен к одному из классов ниже.

Если проблема повторяется, она должна получить:

- durable fix;
- regression test или semantic gate;
- статус закрытия;
- ссылку на audit pack.

## Статусы

- `open` - класс подтвержден, но системно не исправлен.
- `in_progress` - фикс реализуется.
- `fixed` - есть фикс и regression.
- `accepted_risk` - риск принят явно.
- `test_issue` - проблема в тесте/судье, не в боте.
- `needs_business_decision` - нужен ответ Дмитрия/РОПа.

## Классы

| ID | Class | Status | Описание | Главный фикс | Regression |
|---|---|---|---|---|---|
| FC-001 | `ignored_question` | in_progress | Бот отвечает рядом, но не на прямой вопрос клиента. | Open-question detector + answer-first gate + safe-part-first for multitopic. | Dynamic multi-turn test. |
| FC-002 | `context_loss` | fixed | Бот теряет класс, предмет, формат, цель или предыдущую реплику. | DialogueMemory known slots. | Memory unit/integration tests. |
| FC-003 | `reasked_known_data` | fixed | Бот просит данные, которые клиент или CRM уже дали. | `do_not_ask_again` from DialogueMemory. | Known-data reask tests. |
| FC-004 | `templated_opening` | open | Ответ начинается шаблонно и не реагирует на реплику. | Warm few-shot + template detector. | Template repetition gate. |
| FC-005 | `over_handoff` | in_progress | Менеджер вызывается там, где есть подтвержденный безопасный факт. | Route review + fact-aware answer-first; live-status questions stay manager-checked. | Over-handoff regression. |
| FC-006 | `weak_next_step` | in_progress | Ответ полезный, но не ведет к следующему действию. | One-next-step rule. | Next-step gate. |
| FC-007 | `single_topic_answer_to_multitopic_question` | in_progress | В составном вопросе закрыта только одна безопасная часть. | Multitopic decomposition: safe part first, unsafe/live action second. | Multitopic test. |
| FC-008 | `missing_next_step` | in_progress | Нет ясного следующего шага или вопроса. | One-next-step rule. | Next-step gate. |
| FC-009 | `assumed_unstated_need` | in_progress | Бот приписывает клиенту цель/потребность. | Assumption detector + source/confidence slots; funnel ignores bot answers when extracting client slots. | Assumption regression. |
| FC-010 | `kb_voice` | open | Ответ звучит как выдержка из базы, не как консультант. | Warm examples + rewrite only after facts. | Tone semantic review. |
| FC-011 | `commitment_drift` | fixed | Бот забывает или меняет собственное обещание. | Commitments in DialogueMemory. | Commitment tests. |
| FC-012 | `price_fix_question` | needs_business_decision | Нет сильного сценария "как зафиксировать цену / оформить". | Business rule from Dmitry/ROP. | Price-fix dynamic scenarios. |
| FC-013 | `fact_selection_wrong_scope` | in_progress | Реальный факт выбран не для того продукта/класса/формата/бренда. | Scoped fact selection with current product context. | Fact-scope tests. |
| FC-014 | `stale_test_expectation` | open | Тест ждет старую политику. | Test regrading with decision log. | Test manifest check. |
| FC-015 | `judge_issue` | open | Судья неверно пометил корректный ответ. | Judge rubric patch. | Judge fixture. |
| FC-016 | `business_rule_missing` | needs_business_decision | Требуется решение Дмитрия/РОПа. | Decision log entry. | Manual control until decided. |

## Последние важные выводы

- `DialogueMemory` должен быть структурным состоянием, а не просто большим prompt с историей.
- `v8_targeted16` используется как dev-сигнал, не как честный финальный holdout.
- P0 safety и brand isolation не являются "метриками вкуса"; это hard gates.
- LLM-rewriter допустим только под feature flag и только после повторных guards.

## Реализация 2026-05-23

- Добавлен `DialogueMemory` для текущей Telegram-сессии: известные слоты, открытый вопрос, последние ходы, обязательства бота, флаги риска, история маршрутов.
- `DialogueMemory` подключен к сборке `PilotContext`, публичному Telegram-пилоту, динамическому симулятору, `answer_quality_rewriter` и локальному store.
- В динамических транскриптах теперь сохраняется `bot_dialogue_memory` на каждом ходе.
- Добавлен skill `bot-failure-class-review`, чтобы каждый FAIL/PASS_WITH_NOTES превращался в класс проблемы, а не в точечную правку фразы.
- Базовые regression-тесты закрывают память слотов, запрет повторного запроса известных данных, сохранение обязательств и наличие памяти в симуляторе.

## Итерация v8 targeted / ЛВШ Фотон 2026-05-23

- `FC-013 fact_selection_wrong_scope`: исправлен выбор факта между городским лагерем, ЛВШ Менделеево, регулярными онлайн/очными курсами. Добавлены regression-тесты на онлайн-цену против ЛВШ, выездной лагерь и follow-up по транспорту.
- `FC-009 assumed_unstated_need`: исправлена самозагрязняющаяся воронка, где предмет мог вытягиваться из ответов бота (`Python/программирование`), а не из слов клиента. Добавлен тест `test_funnel_slots_ignore_bot_answers_to_avoid_self_pollution`.
- `FC-007 single_topic_answer_to_multitopic_question`: добавлено правило для вопросов вида «трансфер есть? и можно закрепить место?» — сначала ответить на безопасную часть по факту, затем отдельно не обещать место/бронь и передать менеджеру.
- `FC-005 over_handoff`: live-статус мест/группы/смены вынесен в отдельное правило: бот может полезно ответить, но не должен обещать наличие и не должен переписывать этот ответ LLM-полировкой.
- Контрольный сценарий `v8_foton_t10_camp_02`: было `FAIL`/fabrication из-за «программирование», после фиксов `PASS_WITH_NOTES`, hard gates 0, tone 72. Остался мягкий `ignored_question` на раннем ходе: выездной формат с проживанием раскрывается не сразу полностью.

## Итерация Memory 2.0 / ConversationIntentPlan 2026-05-24

- `FC-001 ignored_question`: добавлен `ConversationIntentPlan`, который фиксирует прямой вопрос клиента, известные слоты, нужные факты и следующий шаг. Отдельные слова теперь считаются сигналами, а не единственным основанием маршрута.
- `FC-013 fact_selection_wrong_scope`: исправлена протечка логики УНПК "онлайн-фрагмент" в Фотон. Для Фотона используется "онлайн-пробное", для УНПК - "онлайн-фрагмент"; добавлены regression-тесты на брендовый trial-process.
- `FC-013 fact_selection_wrong_scope`: слово "записаться" больше не трактуется как "запись урока"; вопрос "живое пробное с преподавателем или запись?" больше не уводит в общий шаблон про преподавателей.
- `FC-011 commitment_drift`: неподтверждённые сроки ответа менеджера ("завтра", "до вечера", "в течение суток") блокируются общим guard-ом. Добавлен regression-тест.
- `FC-006 weak_next_step`: добавлены безопасные процессные ответы для "передайте менеджеру", "как получить фрагмент/пробное", "какие данные нужны", "как закрепить цену".
- `FC-003 reasked_known_data`: process-ответы теперь явно показывают уже известные класс/предмет/формат и не просят повторять их.
- Контрольный targeted16 после фиксов: 16/16 без hard-gate FAIL, `PASS_WITH_NOTES` по всем диалогам; средний tone 55.7. Открыты мягкие классы `templated_opening`, `ignored_question`, `over_handoff`.

## Итерация D1 Last Layer / Clean Holdout 2026-05-24

Audit pack: `audits/_inbox/telegram_d1_holdout_clean_after_punchlist_20260524_164854`

- `FC-001 ignored_question`: targeted16 улучшен с 10 до 5 soft-флагов по судье, но clean holdout всё ещё показал 20 `ignored_question`.
- `FC-004 templated_opening`: targeted16 почти не улучшился (10 -> 9); clean holdout показал 11 случаев. Класс остаётся open.
- `FC-005 over_handoff`: targeted16 ухудшился 5 -> 6; clean holdout показал 13 случаев. Нужна более точная граница "факт есть" vs "нужна live-проверка".
- `FC-009 assumed_unstated_need`: исправлено ложное извлечение предмета `программирование` из слова `программа`; добавлены regression-тесты. Clean holdout всё ещё нашёл широкий случай "точные предметы" -> "программирование" в first-pass generation, класс остаётся in_progress.
- `FC-013 fact_selection_wrong_scope`: исправлен online/offline trial context через единый `known_context_fields`; clean holdout открыл новые scope-классы: matkap vs tax, class schedule vs office hours, day camp vs residential LVSH.
- `FC-017 p0_mid_dialog_recall` (new, open): P0-сообщение после безопасного pricing-flow не должно наследовать продажный сценарий. Регрессия: `price -> double charge/refund -> complaint`.
- `FC-018 matkap_tax_confusion` (new, open): follow-up по СФР/документам в контексте маткапитала не должен уходить в налоговый вычет/ФНС.
- `FC-019 schedule_office_hours_confusion` (new, open): график работы/контактов нельзя использовать как расписание занятий группы.
- `FC-020 camp_day_vs_residential_scope` (new, open): явное "без проживания / дневной формат" блокирует шаблон выездной ЛВШ с проживанием.

Замер:

- Unit/profile: 242/242.
- Targeted16 dev: FAIL 0, hard-gate 0, PASS 1, PASS_WITH_NOTES 15, tone 67.6.
- Clean holdout: FAIL 5, hard-gate 4, PASS 4, PASS_WITH_NOTES 17, tone 60.4.

Решение: clean holdout не тюним в этом цикле. Следующий цикл должен писать новые dev-сценарии под классы FC-017..FC-020 и только после этого запускать новый независимый holdout.

## Итерация D1 Root Fixes B/A/C 2026-05-24

Основание: `Claude Cowork Space/02_bot_quality/D1_TZ_root_fixes_holdout_2026-05-24.md`.

- `FC-017 p0_mid_dialog_recall`: реализован общий P0-latch в `DialogueMemory`. После реального P0/спора оплаты диалог остаётся `manager_only` до явного события менеджера; P0 проверяется на каждом ходе. Добавлены регрессии на двойное списание, mid-dialog P0, сохранение latch и benign-гипотетический возврат.
- `FC-018 matkap_tax_confusion`: добавлен `fact_scope=matkap_process/tax_deduction` и `blocked_neighbor_scopes`. Соседний факт про ФНС/налоговый вычет не попадает в `confirmed_facts` для вопроса про маткапитал и СФР.
- `FC-019 schedule_office_hours_confusion`: добавлен `fact_scope=class_schedule/office_hours`. Часы работы и контакты больше не считаются расписанием занятий.
- `FC-020 camp_day_vs_residential_scope`: добавлен `fact_scope=city_day_camp/residential_lvsh`. Дневной/городской формат без проживания не получает факты выездной ЛВШ с проживанием.
- `FC-001 ignored_question` / `FC-004 templated_opening`: прямое улучшение частично зависит от first-pass генерации, но анти-повтор и answer-contract теперь не должны переиспользовать старые зелёные шаблоны вместо ответа на дельту вопроса.

Проверки:

- Root unit set: 213/213.
- D1 bot-quality unit set: 269/269.
- Full collect-only: 1871 tests collected.

Статус: формальные регрессии закрыты. Нужен новый независимый holdout от Claude для честного смыслового замера; старый holdout не использовать для тюнинга.
