# DIFF KB v6.5 -> v6.6 client-safe texts

Основание: `kb_v6_6_staging.md`, решения Дмитрия 2026-06-06/07/08; сборка через `scripts/build_kb_release_v6_1_team_answers.py`.

## Summary

- old client-safe facts: `597`
- new client-safe facts: `607`
- added client-safe fact keys: `22`
- removed from client-safe: `12`
- changed client-safe texts: `7`

## Added client-safe

| brand | fact_key | client_safe_text |
| --- | --- | --- |
| foton | kb_v6_6_client_safe_facts_2026_06_08.attendance_makeup_other_group.client_safe_text | Фотон: пропущенное занятие можно отработать в другой группе, подходящей по времени и уровню; тема занятия может отличаться. Справка для отработки не обязательна. |
| foton | kb_v6_6_client_safe_facts_2026_06_08.attendance_online_instead_offline_exception_2026_27.client_safe_text | Фотон: подключиться к онлайн-занятию вместо очного нельзя; исключение — пропуск 4 и более занятий подряд по болезни. |
| foton | kb_v6_6_client_safe_facts_2026_06_08.entrance_test_results_delivery.client_safe_text | Фотон: результаты вступительного теста появляются в той же тестовой форме в течение недели; по запросу вышлем их вручную. |
| foton | kb_v6_6_client_safe_facts_2026_06_08.homework_check_always_checked.client_safe_text | Фотон: домашние задания всегда проверяются. |
| foton | kb_v6_6_client_safe_facts_2026_06_08.materials_handouts_default.client_safe_text | Фотон: раздаточных конспектов по умолчанию нет: есть чат с преподавателем для домашних заданий и материалов, на части курсов — методички. |
| foton | kb_v6_6_client_safe_facts_2026_06_08.offline_first_lesson_diagnostic_check.client_safe_text | Фотон: на первом очном занятии возможна устная беседа или небольшая контрольная; с собой лучше взять решения вступительного теста. |
| foton | kb_v6_6_client_safe_facts_2026_06_08.offline_groups_levels.client_safe_text | Фотон: очные группы делятся по уровням: базовый, продвинутый и олимпиадный. Разница — в темпе, сложности задач и времени на повторение. |
| foton | kb_v6_6_client_safe_facts_2026_06_08.offline_phys_tech_no_separate_groups_foton_2026_27.client_safe_text | Фотон: отдельных очных групп «Физтех» нет — олимпиадные группы идут внутри общего расписания. Продвинутые и базовые группы отличаются темпом, сложностью задач и временем на повторение; планы по темам почти одинаковые. |
| foton | kb_v6_6_client_safe_facts_2026_06_08.schedule_level_split_parallel_groups.client_safe_text | Фотон: если по расписанию написано «уточняется», это потому что группы делятся по уровню знаний. Возможно, предложим другую группу; постараемся сохранить удобное время, потому что групп в параллели много. |
| foton | kb_v6_6_client_safe_facts_2026_06_08.study_pause_flexible_schedule_personal.client_safe_text | Фотон: при личных обстоятельствах можно заморозить обучение или подобрать гибкий график. |
| foton | kb_v6_6_client_safe_facts_2026_06_08.trial_other_group_free_trial.client_safe_text | Фотон: при сомнениях можно бесплатно посетить пробное занятие в другой группе. |
| unpk | kb_v6_6_client_safe_facts_2026_06_08.annual_online_courses_math_physics_5_11_weekend_2026_27.client_safe_text | УНПК МФТИ: годовые онлайн-курсы по математике и физике для 5–11 классов проходят по выходным: 35 занятий в год, 1 занятие в неделю по 3 ак.ч. Стоимость: семестр — 37 000 ₽, год — 59 000 ₽. |
| unpk | kb_v6_6_client_safe_facts_2026_06_08.annual_online_courses_math_physics_informatics_9_11_weekday_2026_27.client_safe_text | УНПК МФТИ: онлайн-курсы для 9 и 11 классов по будням (математика, физика, информатика), 2 занятия в неделю по 2 ак.ч. Стоимость на 2026/27: семестр — 41 800 ₽, год — 69 900 ₽. |
| unpk | kb_v6_6_client_safe_facts_2026_06_08.attendance_makeup_other_group.client_safe_text | УНПК МФТИ: пропущенное занятие можно отработать в другой группе, подходящей по времени и уровню; тема занятия может отличаться. Справка для отработки не обязательна. |
| unpk | kb_v6_6_client_safe_facts_2026_06_08.attendance_online_instead_offline_exception_2026_27.client_safe_text | УНПК МФТИ: подключиться к онлайн-занятию вместо очного нельзя; исключение — пропуск 4 и более занятий подряд по болезни. |
| unpk | kb_v6_6_client_safe_facts_2026_06_08.homework_check_always_checked.client_safe_text | УНПК МФТИ: домашние задания всегда проверяются. |
| unpk | kb_v6_6_client_safe_facts_2026_06_08.materials_handouts_default.client_safe_text | УНПК МФТИ: раздаточных конспектов по умолчанию нет: есть чат с преподавателем для домашних заданий и материалов, на части курсов — методички. |
| unpk | kb_v6_6_client_safe_facts_2026_06_08.offline_courses_seminar_format.client_safe_text | УНПК МФТИ: очные занятия проходят в семинарском формате — теория и практика вместе, разбор задач у доски, самостоятельная работа. |
| unpk | kb_v6_6_client_safe_facts_2026_06_08.offline_groups_levels.client_safe_text | УНПК МФТИ: очные группы делятся по уровням: базовый, продвинутый и олимпиадный. Разница — в темпе, сложности задач и времени на повторение. |
| unpk | kb_v6_6_client_safe_facts_2026_06_08.offline_phys_tech_level_default.client_safe_text | УНПК МФТИ: очная подготовка к Физтеху отдельных уровней не имеет — это олимпиадный уровень по умолчанию. |
| unpk | kb_v6_6_client_safe_facts_2026_06_08.schedule_level_split_parallel_groups.client_safe_text | УНПК МФТИ: если по расписанию написано «уточняется», это потому что группы делятся по уровню знаний. Возможно, предложим другую группу; постараемся сохранить удобное время, потому что групп в параллели много. |
| unpk | kb_v6_6_client_safe_facts_2026_06_08.study_pause_flexible_schedule_personal.client_safe_text | УНПК МФТИ: при личных обстоятельствах можно заморозить обучение или подобрать гибкий график. |

## Removed from client-safe / gated

| brand | fact_key | old_client_safe_text | new_status_or_text |
| --- | --- | --- | --- |
| foton | individual_lessons_foton.format | Фотон: индивидуальные занятия — Только онлайн, 1-на-1. | not client-safe; route=draft_for_manager; reasons=blocked_status:do_not_use,not_client_allowed_status:do_not_use |
| foton | individual_lessons_foton.prices.lesson_45min | Фотон: индивидуальные занятия, занятие 45 минут — 3 900 ₽. | not client-safe; route=bot_answer_self_for_pilot; reasons=blocked_status:do_not_use,not_client_allowed_status:do_not_use |
| foton | individual_lessons_foton.prices.package_5_sessions | Фотон: индивидуальные занятия, пакет 5 занятий — 23 000 ₽. | not client-safe; route=bot_answer_self_for_pilot; reasons=blocked_status:do_not_use,not_client_allowed_status:do_not_use |
| foton | individual_lessons_foton.prices.session_90min | Фотон: индивидуальные занятия, занятие 90 минут — 6 900 ₽. | not client-safe; route=bot_answer_self_for_pilot; reasons=blocked_status:do_not_use,not_client_allowed_status:do_not_use |
| foton | prices_regular_2026_27.early_booking_2026_27.benefit | Фотон: цены на 2026/27 учебный год — Зафиксировать цену + выбрать день недели. | not client-safe; route=bot_answer_self_for_pilot; reasons=blocked_status:do_not_use,not_client_allowed_status:do_not_use |
| foton | prices_regular_2026_27.early_booking_2026_27.description | Фотон: цены на 2026/27 учебный год — Раннее бронирование для действующих учеников Фотона. | not client-safe; route=bot_answer_self_for_pilot; reasons=blocked_status:do_not_use,not_client_allowed_status:do_not_use |
| foton | prices_regular_2026_27.early_booking_2026_27.online_5_11_semester | Фотон: цены на 2026/27 учебный год, 5-11 класс, онлайн — 29 750 ₽. | not client-safe; route=bot_answer_self_for_pilot; reasons=blocked_status:do_not_use,not_client_allowed_status:do_not_use |
| foton | prices_regular_2026_27.early_booking_2026_27.online_5_11_year | Фотон: цены на 2026/27 учебный год, 5-11 класс, онлайн — 47 250 ₽. | not client-safe; route=bot_answer_self_for_pilot; reasons=blocked_status:do_not_use,not_client_allowed_status:do_not_use |
| foton | prices_regular_2026_27.early_booking_2026_27.payment_deadline | Фотон: цены на 2026/27 учебный год — 15 мая 2026. | not client-safe; route=draft_for_manager; reasons=blocked_status:do_not_use,not_client_allowed_status:do_not_use |
| foton | prices_regular_2026_27.early_booking_2026_27.signup_deadline | Фотон: цены на 2026/27 учебный год — 1 мая 2026. | not client-safe; route=draft_for_manager; reasons=blocked_status:do_not_use,not_client_allowed_status:do_not_use |
| unpk | locations_unpk.free_trial_offline.available | УНПК: адрес и место занятий, очно — нет. | not client-safe; route=draft_for_manager; reasons=blocked_status:do_not_use,not_client_allowed_status:do_not_use |
| unpk | locations_unpk.free_trial_offline.offer | УНПК: по бесплатному пробному очному занятию менеджер подскажет доступный вариант; лимит и возможность записи проверяются с куратором филиала. | not client-safe; route=draft_for_manager; reasons=blocked_status:do_not_use,not_client_allowed_status:do_not_use |

## Changed client-safe text

| brand | fact_key | old_client_safe_text | new_client_safe_text |
| --- | --- | --- | --- |
| foton | installment.services.dolyami.applies_to_all_foton_products | Фотон: рассрочка и оплата — да. | Фотон: рассрочка и оплата — Долями можно рассмотреть для продуктов Фотона; менеджер проверит условия по выбранной программе. |
| foton | installment.services.dolyami.available | Фотон: рассрочка и оплата — да. | Фотон: рассрочка и оплата — Сервис Долями доступен как вариант оплаты; условия и оформление проверяет менеджер. |
| foton | installment.services.dolyami.products_scope_client_safe | Фотон: рассрочка и оплата — Продукты Фотона: очные и онлайн-курсы, ЛВШ, ЛШ и другие программы. | Фотон: рассрочка и оплата — Долями и оплату частями можно рассмотреть для очных и онлайн-курсов, ЛВШ, ЛШ и других программ Фотона; условия оформляет менеджер. |
| foton | installment.services.rassrochka.applies_to_all_foton_products | Фотон: рассрочка и оплата — да. | Фотон: рассрочка и оплата — Оплата частями доступна для очных и онлайн-курсов, ЛВШ, ЛШ и других программ Фотона; конкретные условия и оформление проверяет менеджер. |
| unpk | prices_regular_2026_27.online_olympiad_phystech_classes.client_safe_text | Олимпиадная подготовка Физтех онлайн — для 9 и 11 классов; по другим классам возможность группы уточнит менеджер. | Олимпиадная подготовка онлайн идёт в рамках обычных онлайн-курсов, в группах олимпиадного уровня; отдельного продукта нет. |
| unpk | prices_regular_2026_27.online_olympiad_phystech_classes.product | УНПК: цены на 2026/27 учебный год, онлайн, продукт — Олимпиадная подготовка Физтех онлайн. | УНПК: цены на 2026/27 учебный год, онлайн, продукт — Олимпиадный уровень внутри обычных онлайн-курсов. |
| unpk | team_answers.q15.unpk_online_other_classes.manager_handoff | По онлайн-направлениям УНПК вне подтверждённого формата 2 раза в неделю для 5-11 классов точные условия должен проверить менеджер. | Вне двух подтверждённых онлайн-тарифов УНПК точные условия проверяет менеджер. |

## Focused safety checks

- stale_early_booking_deadline_removed: `0` client-safe hits
- individual_lesson_prices_client_safe: `0` client-safe hits
- degenerate_installment_yes_no: `0` client-safe hits
- unpk_free_trial_offline_client_safe: `0` client-safe hits

## quality_report.json

- quality_passed: `True`
- blocking_failures: `[]`
- client_allowed_facts: `607`

## semantic_review.json

- semantic_pass: `True`
- blocking_findings: `0`
- facts_total: `982`
