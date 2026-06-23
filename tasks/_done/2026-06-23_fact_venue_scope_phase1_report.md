# Fact Venue Scope Phase 1 — Data Markup

Дата: 2026-06-23
Ветка: codex/fact-venue-scope
База: `product_data/knowledge_base/kb_release_20260612_v6_7_staging_r4_1`

## Что сделано

Размечены две независимые оси у scope-фактов базы знаний:

- `venue`: `moscow_regular`, `dolgoprudny`, `lvsh_mendeleevo`, `online`, `any`.
- `program_kind`: `regular`, `camp_city`, `camp_lvsh`, `olympiad`, `any`.

`any` означает общий факт, который нельзя вырезать по площадке: контакты, общий календарь, общие правила, служебные хвосты без структурной привязки к одной площадке.

## Метод вывода venue

Жёсткое правило соблюдено: `venue` выводился только из структуры факта:

- `fact_key`: адресные индексы `locations_unpk.addresses.N.*`, `locations_foton.addresses.N.*`, `schedule_2026_27.*`, `academic_year_2026_27.start_by_location.*`, ключи `patsayeva`, `mfti_institutsky`, `sretenka`, `krasnoselskaya`, `moscow`, `online`, `onlajn`, `lvsh`, `mendeleevo`.
- `product`: `lvsh_mendeleevo_2026`, `lvsh_2026`, `zvsh_mendeleevo`, `regular_online_courses_2026_27`, `m9_online_math_oge`, `m11_online_math_ege`, городские лагеря и регулярные продукты.
- `structured_value`: только структурное поле `format=online` и `path`, если оно дублирует `fact_key`.
- `brand` использован только вместе с `product/format` для структурного случая Фотон-очно: `regular_courses_2026_27` + `brand=foton` + `offline` => `moscow_regular`.

`client_safe_text`, `fact_text`, `raw_value` и любые человекочитаемые тексты для вывода venue не использовались.

## Счётчики

- Всего размечено scope-фактов: 400 из 1075.
- По venue: lvsh_mendeleevo=119, moscow_regular=97, any=90, online=78, dolgoprudny=16
- По program_kind: regular=171, camp_lvsh=117, camp_city=45, any=37, olympiad=30
- Источник venue: product_or_fact_key=117, manual_any_structural_tail=90, fact_key=84, structured_value.format_or_fact_key=72, product_brand_fact_key=13, fact_key.address_index=12, product=6, product_brand_format=4, product_brand=2
- Источник program_kind: product_or_fact_key=192, product_or_fact_type=171, manual_any_structural_tail=37
- По брендам: unpk=220, foton=179, internal=1

## Продукты в разметке

- lvsh_mendeleevo_2026: 108
- schedule_2026_27: 107
- regular_courses_2026_27: 42
- city_camp_2026: 39
- kb_v6_6_client_safe_facts_2026_06_08: 21
- academic_year_2026_27: 14
- locations_unpk: 14
- contacts_foton: 8
- city_summer_school_2026: 6
- locations_foton: 6
- contacts_unpk: 6
- online_olympiad_phystech_classes_9_11: 6
- lvsh_2026: 4
- fiztech_olympiad_general: 3
- zvsh_mendeleevo: 3
- boarding_and_camps: 2
- contacts_and_visits: 2
- m11_online_math_ege: 2
- m9_online_math_oge: 2
- trial_and_acquaintance: 2
- regular_online_courses_2026_27: 2
- main_office: 1

## Сэмпл 20 фактов

| brand | product | fact_key | venue | program_kind | venue_source | program_source |
|---|---|---|---|---|---|---|
| foton | academic_year_2026_27 | `academic_year_2026_27.start_by_location.moscow` | moscow_regular | regular | fact_key | product_or_fact_type |
| foton | city_camp_2026 | `ls_city_2026_foton.discounts.already_paid_2026_27` | moscow_regular | camp_city | product_brand_fact_key | product_or_fact_key |
| foton | schedule_2026_27 | `schedule_2026_27.groups.group_start_date_i133_krasnoselskaya_sun_1215_1335_math_3_4_olympiad_before_2027_05_23.client_safe_text` | moscow_regular | olympiad | fact_key | product_or_fact_key |
| foton | locations_foton | `locations_foton.addresses.1.address` | moscow_regular | any | fact_key.address_index | manual_any_structural_tail |
| unpk | academic_year_2026_27 | `academic_year_2026_27.start_by_location.mfti_institutsky_9` | dolgoprudny | regular | fact_key | product_or_fact_type |
| unpk | city_camp_2026 | `ls_city_2026_unpk.dolgoprudny.location` | dolgoprudny | camp_city | fact_key | product_or_fact_key |
| unpk | locations_unpk | `locations_unpk.addresses.2.address` | dolgoprudny | any | fact_key.address_index | manual_any_structural_tail |
| foton | boarding_and_camps | `r4_1_owner_2026_06_12.foton.no_boarding_except_lvsh` | lvsh_mendeleevo | camp_lvsh | product_or_fact_key | product_or_fact_key |
| unpk | locations_unpk | `locations_unpk.addresses.4.address` | lvsh_mendeleevo | any | fact_key.address_index | manual_any_structural_tail |
| foton | academic_year_2026_27 | `academic_year_2026_27.start_by_location.online` | online | regular | structured_value.format_or_fact_key | product_or_fact_type |
| unpk | online_olympiad_phystech_classes_9_11 | `prices_regular_2026_27.online_olympiad_phystech_9_and_11.bot_behavior_when_asked` | online | olympiad | structured_value.format_or_fact_key | product_or_fact_key |
| unpk | locations_unpk | `locations_unpk.online_trial_fragment.available` | online | any | structured_value.format_or_fact_key | manual_any_structural_tail |
| foton | academic_year_2026_27 | `academic_year_2026_27.semester_1_weeks` | any | regular | manual_any_structural_tail | product_or_fact_type |
| unpk | city_camp_2026 | `ls_city_2026_unpk.discounts.already_paid_2026_27` | any | camp_city | manual_any_structural_tail | product_or_fact_key |
| unpk | fiztech_olympiad_general | `fiztech_olympiad.open_question` | any | olympiad | manual_any_structural_tail | product_or_fact_key |
| foton | contacts_and_visits | `r4_owner_2026_06_11.foton.center_visit_by_appointment` | any | any | manual_any_structural_tail | manual_any_structural_tail |
| foton | academic_year_2026_27 | `academic_year_2026_27.semester_2_weeks` | any | regular | manual_any_structural_tail | product_or_fact_type |
| foton | academic_year_2026_27 | `academic_year_2026_27.start` | any | regular | manual_any_structural_tail | product_or_fact_type |
| foton | academic_year_2026_27 | `academic_year_2026_27.total_lessons` | any | regular | manual_any_structural_tail | product_or_fact_type |
| foton | city_camp_2026 | `ls_city_2026_foton.discounts.former_student_same_brand` | moscow_regular | camp_city | product_brand_fact_key | product_or_fact_key |

## Список any

Полный список: `any_facts.csv`.

Первые 40 строк:

- `academic_year_2026_27.semester_1_weeks` (foton, academic_year_2026_27, regular)
- `academic_year_2026_27.semester_1_weeks` (unpk, academic_year_2026_27, regular)
- `academic_year_2026_27.semester_2_weeks` (foton, academic_year_2026_27, regular)
- `academic_year_2026_27.semester_2_weeks` (unpk, academic_year_2026_27, regular)
- `academic_year_2026_27.start` (foton, academic_year_2026_27, regular)
- `academic_year_2026_27.start` (unpk, academic_year_2026_27, regular)
- `academic_year_2026_27.total_lessons` (foton, academic_year_2026_27, regular)
- `academic_year_2026_27.total_lessons` (unpk, academic_year_2026_27, regular)
- `ls_city_2026_unpk.discounts.already_paid_2026_27` (unpk, city_camp_2026, camp_city)
- `ls_city_2026_unpk.discounts.former_student_same_brand` (unpk, city_camp_2026, camp_city)
- `ls_city_2026_unpk.discounts.mfti_employees` (unpk, city_camp_2026, camp_city)
- `ls_city_2026_unpk.discounts.multichild` (unpk, city_camp_2026, camp_city)
- `ls_city_2026_unpk.discounts.refer_a_friend.advanced` (unpk, city_camp_2026, camp_city)
- `ls_city_2026_unpk.discounts.refer_a_friend.base` (unpk, city_camp_2026, camp_city)
- `ls_city_2026_unpk.free_morning_club.from_day` (unpk, city_camp_2026, camp_city)
- `ls_city_2026_unpk.free_morning_club.name` (unpk, city_camp_2026, camp_city)
- `ls_city_2026_unpk.free_morning_club.time` (unpk, city_camp_2026, camp_city)
- `ls_city_2026_unpk.meals` (unpk, city_camp_2026, camp_city)
- `ls_city_2026_unpk.not_to_confuse_with` (unpk, city_camp_2026, camp_city)
- `r4_1_owner_2026_06_12.unpk.city_summer_school_tariffs` (unpk, city_summer_school_2026, camp_city)
- `r4_owner_2026_06_11.foton.center_visit_by_appointment` (foton, contacts_and_visits, any)
- `r4_owner_2026_06_11.unpk.center_visit_by_appointment` (unpk, contacts_and_visits, any)
- `contacts_foton.contact_hours.client_safe_text` (foton, contacts_foton, any)
- `contacts_foton.email` (foton, contacts_foton, any)
- `contacts_foton.phone` (foton, contacts_foton, any)
- `contacts_foton.telegram` (foton, contacts_foton, any)
- `contacts_foton.toll_free` (foton, contacts_foton, any)
- `contacts_foton.vk` (foton, contacts_foton, any)
- `contacts_foton.website` (foton, contacts_foton, any)
- `contacts_foton.whatsapp` (foton, contacts_foton, any)
- `contacts_unpk.contact_hours.client_safe_text` (unpk, contacts_unpk, any)
- `contacts_unpk.email` (unpk, contacts_unpk, any)
- `contacts_unpk.phone` (unpk, contacts_unpk, any)
- `contacts_unpk.telegram` (unpk, contacts_unpk, any)
- `contacts_unpk.toll_free` (unpk, contacts_unpk, any)
- `contacts_unpk.website` (unpk, contacts_unpk, any)
- `fiztech_olympiad.open_question` (unpk, fiztech_olympiad_general, olympiad)
- `fiztech_olympiad.prices.group` (unpk, fiztech_olympiad_general, olympiad)
- `fiztech_olympiad.prices.individual` (unpk, fiztech_olympiad_general, olympiad)
- `kb_v6_6_client_safe_facts_2026_06_08.attendance_makeup_other_group.client_safe_text` (foton, kb_v6_6_client_safe_facts_2026_06_08, regular)
