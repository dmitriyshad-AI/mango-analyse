# KB v6.4 — Артефакт C: manager_only факты, застрявшие не на месте

Автор: второй Claude (ночь, Задача 2). Источник: `manager_only_or_internal_facts.jsonl` (363 факта). Новый артефакт из находок приложения.

**Разумное умолчание (эвристика should_move=true):** (а) `route_policy==bot_answer_self_for_pilot` (бот уже допущен, но факт лежит в manager_only), ИЛИ (б) фактологический client-релевантный факт (цена/дата/формат/контакт/справка) типа price/course_parameter/deadline/camp/contact без policy-блокера. Блокеры: promocode, teacher (ФИО), refund/претензии, лицензии/юр.данные, note_internal, risk high. Устаревшие (`do_not_use_as_current_price`) помечены «требует подтверждения Дмитрия».

Пройдено всех manager_only: 363. Кандидатов на перенос (true): 35. Оставить в manager_only (false): 328.

## Главный пример из находок

Онлайн-цена УНПК (41 800/69 900) — `prices_regular_2026_27.online_olympiad_phystech_9_and_11.note_internal` — застряла в manager_only с пометкой «прошлый год». Клиенты прямо спрашивают онлайн-цену (`C2_multi_subj_03`, `C1_format_price_03` — оба FAIL по этой причине). Должна быть в client-safe ПОСЛЕ актуализации суммы командой.

## Кандидаты на перенос в client-safe (should_move=true)

| manager_only_fact_key | brand | preview | fact_type | route_policy | should_move | updated_text_proposal / reason |
|---|---|---|---|---|---|---|
| `team_answers.q11.modular_courses_m9_m11.old_pr` | foton | Устаревший верхний край цены модульных М9/М11: 94 500  | price | manager_handoff_only | true | ТРЕБУЕТ ПОДТВЕРЖДЕНИЯ ДМИТРИЯ: значение помечено do_not_use_as_current_price (возможно неа |
| `team_answers.q11.modular_courses_m9_m11.old_pr` | foton | Устаревший нижний край цены модульных М9/М11: 18 900 ₽ | price | manager_handoff_only | true | ТРЕБУЕТ ПОДТВЕРЖДЕНИЯ ДМИТРИЯ: значение помечено do_not_use_as_current_price (возможно неа |
| `fiztech_olympiad.prices.group` | unpk | УНПК: prices , group — 50 000 ₽. | price | manager_handoff_only | true | УНПК: prices , group — 50 000 ₽. |
| `fiztech_olympiad.prices.individual` | unpk | УНПК: prices , individual — 33 000 ₽. | price | manager_handoff_only | true | УНПК: prices , individual — 33 000 ₽. |
| `team_answers.q15.unpk_online_other_classes.man` | unpk | По онлайн-направлениям УНПК вне олимпиадной подготовки | price | manager_handoff_only | true | По онлайн-направлениям УНПК вне олимпиадной подготовки Физтех для 9 и 11 классов точные ус |
| `prices_regular_2026_27.online_olympiad_phystec` | unpk | УНПК: цены на 2026/27 учебный год, 9 и 11 класс, онлай | price | bot_answer_self_for_pi | true | УНПК: цены на 2026/27 учебный год, 9 и 11 класс, онлайн — По онлайн-курсам УНПК на 2026/27 |
| `prices_regular_2026_27.online_olympiad_phystec` | unpk | УНПК: цены на 2026/27 учебный год, 9 и 11 класс, онлай | price | bot_answer_self_for_pi | true | УНПК: цены на 2026/27 учебный год, 9 и 11 класс, онлайн, продукт — Будничные онлайн-курсы  |
| `prices_regular_2026_27.online_5_11_class_regul` | unpk | УНПК: цены на 2026/27 учебный год, 5-11 класс, онлайн  | price | bot_answer_self_for_pi | true | УНПК: цены на 2026/27 учебный год, 5-11 класс, онлайн — Да, онлайн-направление есть, но ак |
| `certificates.types.course_attendance.lead_time` | foton | Фотон: справки и документы — 10. | course_parameter | manager_handoff_only | true | Фотон: справки и документы — 10. |
| `installment.client_confirmed_terms.client_safe` | foton | В Фотоне можно оплатить обучение частями: доступны вар | installment | bot_answer_self_for_pi | true | В Фотоне можно оплатить обучение частями: доступны варианты на 6, 10 или 12 месяцев, а так |
| `legal_entities.entities.1.used_for.2` | foton | Фотон: used for , 2 — Онлайн Фотон. | documents | manager_handoff_only | true | Фотон: used for , 2 — Онлайн Фотон. |
| `legal_entities.entities.1.used_for.4` | foton | Фотон: used for , 4 — Рассрочка Т-Банк. | documents | manager_handoff_only | true | Фотон: used for , 4 — Рассрочка Т-Банк. |
| `legal_entities_full_map.3.used_for.2` | foton | Фотон: used for , 2 — Онлайн Фотон. | documents | manager_handoff_only | true | Фотон: used for , 2 — Онлайн Фотон. |
| `legal_entities_full_map.3.used_for.4` | foton | Фотон: used for , 4 — Рассрочка Т-Банк. | documents | manager_handoff_only | true | Фотон: used for , 4 — Рассрочка Т-Банк. |
| `lvsh_mendeleevo_2026.smeny_2026.1.freshness` | foton | Фотон: даты смен ЛВШ Менделеево требуют ручной проверк | deadline | draft_for_manager | true | Фотон: даты смен ЛВШ Менделеево требуют ручной проверки перед ответом клиенту. |
| `lvsh_mendeleevo_2026.smeny_2026.2.freshness` | foton | Фотон: даты смен ЛВШ Менделеево требуют ручной проверк | deadline | draft_for_manager | true | Фотон: даты смен ЛВШ Менделеево требуют ручной проверки перед ответом клиенту. |
| `team_answers.q11.modular_courses_m9_m11.discon` | foton | Модульных курсов М9/М11 сейчас нет; это были прошлые и | program | manager_handoff_only | true | Модульных курсов М9/М11 сейчас нет; это были прошлые интенсивы. Не предлагать как действую |
| `medical_certificates_validity.certificate_291.` | internal | Внутренне: справки и документы — Срочная справка. | documents | manager_handoff_only | true | Внутренне: справки и документы — Срочная справка. |
| `medical_certificates_validity.certificate_291.` | internal | Внутренне: справки и документы — 3 дня. | documents | manager_handoff_only | true | Внутренне: справки и документы — 3 дня. |
| `medical_certificates_validity.form_079u.purpos` | internal | Внутренне: справки и документы — Для летних выездных л | documents | manager_handoff_only | true | Внутренне: справки и документы — Для летних выездных лагерей. |
| `medical_certificates_validity.form_079u.valid_` | internal | Внутренне: справки и документы — 3 месяца. | documents | manager_handoff_only | true | Внутренне: справки и документы — 3 месяца. |
| `crm_brand_groups.tallanto.heuristics_for_manag` | internal | Внутренне: heuristics for manager , online, онлайн — О | program | manager_handoff_only | true | Внутренне: heuristics for manager , online, онлайн — Онлайн — есть и УНПК и Фотон (скорее  |
| `source_coverage_audit_2026_05_17.critical_unto` | internal | Внутренне: 3 , action — Прочитать выборочно по теме, е | program | manager_handoff_only | true | Внутренне: 3 , action — Прочитать выборочно по теме, если возникнут пробелы. |
| `certificates.types.course_attendance.lead_time` | unpk | УНПК: справки и документы — 10. | course_parameter | manager_handoff_only | true | УНПК: справки и документы — 10. |
| `lvsh_mendeleevo_2026.availability_2026.client_` | unpk | По ЛВШ УНПК места уже почти распроданы; запись сейчас  | camp_lvsh | draft_for_manager | true | По ЛВШ УНПК места уже почти распроданы; запись сейчас только через живого сотрудника. |
| `lvsh_mendeleevo_2026.smeny_2026.1.freshness` | unpk | УНПК: даты смен ЛВШ Менделеево требуют ручной проверки | deadline | draft_for_manager | true | УНПК: даты смен ЛВШ Менделеево требуют ручной проверки перед ответом клиенту. |
| `lvsh_mendeleevo_2026.smeny_2026.2.client_safe_` | unpk | Августовская смена УНПК 15-25 августа закрыта. | deadline | draft_for_manager | true | Августовская смена УНПК 15-25 августа закрыта. |
| `lvsh_mendeleevo_2026.smeny_2026.2.freshness` | unpk | УНПК: даты смен ЛВШ Менделеево требуют ручной проверки | deadline | draft_for_manager | true | УНПК: даты смен ЛВШ Менделеево требуют ручной проверки перед ответом клиенту. |
| `lvsh_mendeleevo_2026.teachers.list.1.subject` | unpk | УНПК: ЛВШ Менделеево — Информатика. | camp_lvsh | manager_handoff_only | true | УНПК: ЛВШ Менделеево — Информатика. |
| `lvsh_mendeleevo_2026.teachers.list.5.subject` | unpk | УНПК: ЛВШ Менделеево — Информатика. | camp_lvsh | manager_handoff_only | true | УНПК: ЛВШ Менделеево — Информатика. |
| `prices_regular_2026_27.online_olympiad_phystec` | unpk | УНПК: цены на 2026/27 учебный год, 9 и 11 класс, онлай | course_parameter | draft_for_manager | true | УНПК: цены на 2026/27 учебный год, 9 и 11 класс, онлайн, классы — 9 и 11. |
| `prices_regular_2026_27.online_olympiad_phystec` | unpk | Олимпиадная подготовка Физтех онлайн сейчас указана дл | course_parameter | draft_for_manager | true | Олимпиадная подготовка Физтех онлайн сейчас указана для 9 и 11 классов; для других классов |
| `prices_regular_2026_27.online_olympiad_phystec` | unpk | УНПК: цены на 2026/27 учебный год, 9 и 11 класс, онлай | course_parameter | draft_for_manager | true | УНПК: цены на 2026/27 учебный год, 9 и 11 класс, онлайн, расписание — будни. |
| `prices_regular_2026_27.online_5_11_class_regul` | unpk | УНПК: цены на 2026/27 учебный год, 5-11 класс, онлайн, | course_parameter | draft_for_manager | true | УНПК: цены на 2026/27 учебный год, 5-11 класс, онлайн, классы — 5-11. |
| `tg_unpk_verified_2026_05_21.manager_only_facts` | unpk | Если ребёнок заболел и есть справка, менеджер подскаже | program | manager_handoff_only | true | Если ребёнок заболел и есть справка, менеджер подскажет варианты по переносу или возврату  |

## Открытые вопросы

1. Все `do_not_use_as_current_price` факты — требуют актуализации суммы командой до переноса (не выдумывать).
2. Часть кандидатов с `route_policy=bot_answer_self_for_pilot` уже фактически разрешена боту — проверить, почему они в manager_only-файле (возможно, ошибка раскладки пакета).
3. Грань «фактологический vs policy» по эвристике — Кодексу/Дмитрию сверить пограничные случаи (документы, intensive).
