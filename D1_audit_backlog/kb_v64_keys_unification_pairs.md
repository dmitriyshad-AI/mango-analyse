# KB v6.4 — Артефакт B: пары одинаковых сущностей с разными ключами (унификация)

Автор: второй Claude (ночь, Задача 2). Внутренний документ для Кодекса (унификация ключей по §1.3/§10.2 плана) — НЕ клиентский текст; бренды сопоставляются только как ключи KB.

**Метод:** автоматическое сопоставление client-safe фактов Foton↔УНПК по похожести `client_safe_text` (Jaccard токенов ≥0.5, текст ≥6 слов, ≥3 общих содержательных токена), где РАЗНЫЕ leaf-ключи. Короткие/генеричные факты («да/нет/число») отфильтрованы. Это кандидаты — Кодексу сверить вручную.

**Находка:** большая часть фактов уже использует ОДИНАКОВЫЕ ключи в обоих брендах (`discounts.second_subject.offline.pct`, `prices_regular_2026_27.*`, `academic_year_2026_27.*`) — KB во многом уже унифицирован. Ниже — именно РАСХОЖДЕНИЯ ключей/индексов.

Найдено пар с разными ключами: 26 (цель ≥20).

| sim | foton_fact_key | foton_text | unpk_fact_key | unpk_text | values_match |
|---|---|---|---|---|---|
| 1.0 | `brand_rules.approved_brand_relationship_an` | Это отдельные организации, по вашему вопросу сориентирую в | `brand_rules.approved_brand_relationship_an` | Это отдельные организации, по вашему вопросу сориентирую в | — |
| 1.0 | `discounts.loyal_customers_lvsh_only.tiers.` | Фотон: скидка постоянным участникам ЛВШ — 10%; условие: по | `discounts.loyal_customers_lvsh_only.tiers.` | УНПК: скидка постоянным участникам ЛВШ — 15%; условие: пос | FALSE |
| 1.0 | `discounts.loyal_customers_lvsh_only.tiers.` | Фотон: скидка постоянным участникам ЛВШ — 15%; условие: по | `discounts.loyal_customers_lvsh_only.tiers.` | УНПК: скидка постоянным участникам ЛВШ — 10%; условие: пос | FALSE |
| 1.0 | `discounts.loyal_customers_lvsh_only.tiers.` | Фотон: скидка постоянным участникам ЛВШ — 20%; условие: по | `discounts.loyal_customers_lvsh_only.tiers.` | УНПК: скидка постоянным участникам ЛВШ — 10%; условие: пос | FALSE |
| 1.0 | `discounts.stacking_rule` | Фотон: если клиенту доступно несколько скидок, они не сумм | `discounts.stacking_rule_text` | УНПК: если клиенту доступно несколько скидок, они не сумми | — |
| 1.0 | `discounts.stacking_rule_text` | Фотон: если клиенту доступно несколько скидок, они не сумм | `discounts.stacking_rule` | УНПК: если клиенту доступно несколько скидок, они не сумми | — |
| 1.0 | `intensives_2026.ege_intensive.available_in` | Фотон: этот интенсив в текущем бренде не проводится. | `intensives_2026.oge_intensive.available_in` | УНПК: этот интенсив в текущем бренде не проводится. | — |
| 1.0 | `bot_policy.approved_phrases.theme_11_contr` | Договор пришлёт менеджер в ближайшие дни на email. Подскаж | `bot_policy.approved_phrases.theme_11_contr` | Договор пришлёт менеджер в ближайшие дни на email. Подскаж | — |
| 1.0 | `bot_policy.approved_phrases.theme_12_certi` | Менеджер подготовит справку и пришлёт в течение 10 дней, п | `bot_policy.approved_phrases.theme_12_certi` | Менеджер подготовит справку и пришлёт в течение 10 дней, п | — |
| 1.0 | `bot_policy.approved_phrases.theme_17_teach` | Преподаватели — из МФТИ, МГУ, ВШЭ, МГТУ им. Баумана, МИФИ. | `bot_policy.approved_phrases.theme_17_teach` | Преподаватели — из МФТИ, МГУ, ВШЭ, МГТУ им. Баумана, МИФИ. | — |
| 0.87 | `objection_responses.too_expensive_course.3` | Фотон: черновик для ситуации «возражение о стоимости курса | `objection_responses.too_expensive_course.5` | УНПК: черновик для ситуации «возражение о стоимости курса» | true |
| 0.86 | `discounts.loyal_customers_lvsh_only.tiers.` | Фотон: скидка постоянным участникам ЛВШ — 30%; условие: по | `discounts.loyal_customers_lvsh_only.tiers.` | УНПК: скидка постоянным участникам ЛВШ — 10%; условие: пос | FALSE |
| 0.75 | `lvsh_mendeleevo_2026.directions.fizmat.sub` | Фотон: в ЛВШ Менделеево для физико-математического направл | `lvsh_mendeleevo_2026.directions.fizmat.sub` | УНПК: в ЛВШ Менделеево для физико-математического направле | — |
| 0.75 | `lvsh_mendeleevo_2026.directions.fizmat.sub` | Фотон: в ЛВШ Менделеево для физико-математического направл | `lvsh_mendeleevo_2026.directions.fizmat.sub` | УНПК: в ЛВШ Менделеево для физико-математического направле | — |
| 0.71 | `academic_year_2026_27.semester_1_weeks` | Фотон: в первом семестре 2026/27 учебного года 20 недель. | `academic_year_2026_27.semester_2_weeks` | УНПК: во втором семестре 2026/27 учебного года 15 недель. | — |
| 0.71 | `academic_year_2026_27.semester_2_weeks` | Фотон: во втором семестре 2026/27 учебного года 15 недель. | `academic_year_2026_27.semester_1_weeks` | УНПК: в первом семестре 2026/27 учебного года 20 недель. | — |
| 0.71 | `discounts.loyal_customers_lvsh_only.not_ap` | Фотон: скидка постоянным участникам ЛВШ не применяется к п | `discounts.loyal_customers_lvsh_only.applie` | УНПК: скидка постоянным участникам ЛВШ применяется к проду | — |
| 0.71 | `discounts.loyal_customers_lvsh_only.not_ap` | Фотон: скидка постоянным участникам ЛВШ не применяется к п | `discounts.loyal_customers_lvsh_only.applie` | УНПК: скидка постоянным участникам ЛВШ применяется к проду | — |
| 0.71 | `online_platform.levels.1` | Фотон: на онлайн-платформе есть уровень обучения: базовый. | `online_platform.levels.2` | УНПК: на онлайн-платформе есть уровень обучения: продвинут | — |
| 0.71 | `online_platform.levels.2` | Фотон: на онлайн-платформе есть уровень обучения: продвину | `online_platform.levels.1` | УНПК: на онлайн-платформе есть уровень обучения: базовый. | — |
| 0.71 | `online_platform.levels.3` | Фотон: на онлайн-платформе есть уровень обучения: олимпиад | `online_platform.levels.1` | УНПК: на онлайн-платформе есть уровень обучения: базовый. | — |
| 0.64 | `objection_responses.brand_link_question.ap` | Фотон: черновик для ситуации «вопрос о связи брендов»: Это | `brand_rules.approved_brand_relationship_an` | Это отдельные организации, по вашему вопросу сориентирую в | — |
| 0.62 | `ls_city_2026_foton.moscow_foton.prices.plu` | Фотон: городской летний лагерь, Москва, расширенный вариан | `ls_city_2026_unpk.moscow.prices.base` | УНПК: городской летний лагерь, Москва, базовый вариант — 3 | FALSE |
| 0.62 | `discounts.loyal_customers_lvsh_only.not_ap` | Фотон: скидка постоянным участникам ЛВШ не применяется к п | `discounts.loyal_customers_lvsh_only.applie` | УНПК: скидка постоянным участникам ЛВШ применяется к проду | — |
| 0.62 | `discounts.loyal_customers_lvsh_only.tiers.` | Фотон: скидка постоянным участникам ЛВШ — 5%; условие: пос | `discounts.loyal_customers_lvsh_only.tiers.` | УНПК: скидка постоянным участникам ЛВШ — 10%; условие: пос | FALSE |
| 0.62 | `objection_responses.too_expensive_course.2` | Фотон: черновик для ситуации «возражение о стоимости курса | `objection_responses.too_expensive_course.1` | УНПК: черновик для ситуации «возражение о стоимости курса» | FALSE |

## Содержательные структурные расхождения (приоритет унификации)

- **Интенсивы:** Foton — `intensives_2026.oge_*` (ОГЭ), УНПК — `intensives_2026.ege_*` (ЕГЭ); индексы `includes.N` различаются. Привести к единой схеме `intensives_2026.<exam>.includes.*`.
- **Очное пробное:** `locations_foton.free_trial_offline_current` (есть) vs `locations_unpk.free_trial_offline.available` (нет) — разные ключи И разные значения; унифицировать ключ, значение оставить разным.
- **Скрипты возражений:** `objection_responses.*.N` — разные индексы N у брендов при одинаковом тексте; унифицировать индексацию.

## Главный пример из плана §1.3

Скидка на второй предмет очно 20%. Если ключ уже совпадает в обоих брендах — унификация выполнена; `other_brand_match=3` судьи возникает из-за ДУБЛЯ факта у УНПК под `payment_options...discount_extra` при наличии `discounts.*` — убрать дубль.

## Открытые вопросы

1. Пороговые пары (sim 0.5-0.6) — приблизительные, ручная сверка.
2. Унифицировать КЛЮЧ, не значение.
3. Проверить дубли фактов у УНПК (discounts.* и payment_options.* об одном).
