# Классификация 167 случаев `false_handoff` (Задача 3)

Автор: второй Claude (ночь). Источник: `dynamic_summary.json` → `over_handoff.false_handoff[]` (167), обогащено `bot_safety_flags`/`route` из `dynamic_dialog_transcripts.jsonl`.

## Ключевая находка (важно для плана)

**ВСЕ 167 случаев имеют `fact_level=retrieved_match`** — то есть факт под вопрос БЫЛ найден, но бот всё равно ушёл в хендофф. Это метрика «факт был, но не доставлен», а не «эскалация без факта».

**P0/high_risk флагов в этих 167 — НОЛЬ.** Значит refund/payment-липкость (Волна 1) в метрику `false_handoff` НЕ попадает вообще: те ходы обнуляют rfk и не дают `retrieved_match`. Волна 1 бьёт по ДРУГОЙ метрике (over-handoff/доля `manager_only` без факта), а не по `false_handoff=167`.

**Вывод-поправка к плану:** чтобы сдвинуть именно число `false_handoff=167`, наибольший рычаг — **Волна 3 (verifier)** и **Волна 2 (редактор-шум)**, НЕ Волна 1. В §4 плана Волна 1 названа «главным источником false_handoff» — для ЭТОЙ метрики это неточно (Волна 1 снижает over-escalation, но не fact-present false_handoff).

## Со-присутствие сигналов (на 167)

- `verification_fallback`/`semantic_fallback`: **99** (59%)
- `unsupported_promise_detected`: **67** (40%)
- какой-либо `*_redacted` (price/discount/percent/…): **83** (50%) — часто со-присутствует с verifier
- ни одного из guard/verifier (чистый «факт не использован», Phase 1): **12**
- route: `manager_only` 70, `draft_for_manager` 97

## Распределение по классам (первичный, по приоритету сигналов)

| класс | случаев | доля | волна |
|---|---|---|---|
| 11.7 | 88 | 53% | 3 |
| 11.3 | 67 | 40% | 3 |
| 11.1 | 9 | 5% | 5 |
| 11.5 | 2 | 1% | 2 |
| Phase1 | 1 | 1% | после волн (вне P11) |

**Ложные защиты Phase 11 (11.1+11.3+11.5+11.7): 166/167 = 99%.** Гипотеза Claude #1 «50-60% ложных защит» — для метрики `false_handoff` ЗАНИЖЕНА: реально ~99% (потому что эта метрика по определению только про ходы с доступным фактом).

## Распределение по волнам

| волна | случаев | доля |
|---|---|---|
| Волна 3 | 155 | 93% |
| Волна 5 | 9 | 5% |
| Волна 2 | 2 | 1% |
| Волна после волн (вне P11) | 1 | 1% |

## Топ-3 диалога по числу false_handoff

- `eval_unpk_C2_multi_subj_02` — 5 ходов
- `eval_foton_C2_multi_subj_01` — 4 ходов
- `eval_foton_C2_multi_subj_02` — 4 ходов

## Полная таблица 167

| dialog_id | turn | brand | client (66) | route | класс | волна |
|---|---|---|---|---|---|---|
| `eval_foton_E1_price_01` | 1 | foton | а это онлайн или очно, и по цене что выходит для 7 класса? | manager_only | 11.3 | 3 |
| `eval_foton_E1_price_03` | 4 | foton | поняла, тогда год онлайн можно долями смотреть | draft_for_manager | 11.7 | 3 |
| `eval_foton_E1_price_04` | 2 | foton | а математика очно для 6 класса сколько стоит? | draft_for_manager | 11.7 | 3 |
| `eval_foton_E1_price_05` | 1 | foton | а это онлайн или очно, и по цене что выходит для 10 класса? | manager_only | 11.3 | 3 |
| `eval_foton_E2_format_record_01` | 2 | foton | а по дням занятий не подскажете? и если пропустит, уроки в записи  | draft_for_manager | 11.7 | 3 |
| `eval_foton_E2_format_record_04` | 1 | foton | здравствуйте, подскажите, а если ребёнок пропустит очное занятие п | draft_for_manager | 11.7 | 3 |
| `eval_foton_E3_schedule_01` | 3 | foton | я про занятия спрашиваю, не про часы связи) в какие дни недели мат | draft_for_manager | 11.7 | 3 |
| `eval_foton_E3_schedule_04` | 2 | foton | ну то есть это в какие дни недели обычно? хотя бы примерно | draft_for_manager | 11.7 | 3 |
| `eval_foton_E4_level_03` | 3 | foton | поняла. а если группа олимпиадная, туда с нуля вообще берут или сн | draft_for_manager | 11.7 | 3 |
| `eval_foton_E4_level_04` | 1 | foton | здравствуйте, сын в 6 классе, математика очно интересует. мы раньш | draft_for_manager | 11.7 | 3 |
| `eval_foton_E5_payment_01` | 1 | foton | а это онлайн или очно, и по цене что выходит для 7 класса? | manager_only | 11.3 | 3 |
| `eval_foton_E5_payment_01` | 3 | foton | а помесячно можно или только семестр/год сразу? | manager_only | 11.3 | 3 |
| `eval_foton_E5_payment_02` | 3 | foton | то есть помесячно у вас нельзя? мне просто надо понять сейчас, я т | draft_for_manager | 11.7 | 3 |
| `eval_foton_E5_payment_02` | 4 | foton | ну понятно, тогда пока не буду оставлять заявку, мне нужен был отв | draft_for_manager | 11.7 | 3 |
| `eval_foton_E5_payment_04` | 2 | foton | а помесячно можно или только сразу за блок? | draft_for_manager | 11.7 | 3 |
| `eval_foton_E5_payment_05` | 1 | foton | здравствуйте, подскажите это онлайн или очно? и по цене что выходи | manager_only | 11.3 | 3 |
| `eval_foton_E5_payment_05` | 3 | foton | а помесячно можно платить или только семестр/год сразу? | manager_only | 11.3 | 3 |
| `eval_foton_E6_discount_01` | 1 | foton | здравствуйте, математика 7 класс онлайн в какие дни проходит и мож | draft_for_manager | 11.7 | 3 |
| `eval_foton_E6_discount_01` | 2 | foton | а по скидкам можете сразу подсказать? какие вообще есть? | draft_for_manager | 11.7 | 3 |
| `eval_foton_E6_discount_05` | 2 | foton | поняла, а какие у вас скидки есть? | draft_for_manager | 11.7 | 3 |
| `eval_foton_E6_discount_05` | 3 | foton | а если у нас в семье двое занимаются, есть какая-то скидка? | manager_only | 11.3 | 3 |
| `eval_foton_E6_discount_05` | 4 | foton | нет, не многодетная. я про то что двое детей у вас учатся, и еще з | draft_for_manager | 11.7 | 3 |
| `eval_foton_E7_olymp_01` | 2 | foton | а вы готовите к олимпиадам по математике? | draft_for_manager | 11.7 | 3 |
| `eval_foton_E7_olymp_03` | 3 | foton | я про другое спрашиваю, к олимпиадным задачам по информатике можно | manager_only | 11.5 | 2 |
| `eval_foton_E7_olymp_04` | 3 | foton | поняла. а к олимпиадным задачам реально можно прокачаться в 6 клас | draft_for_manager | 11.7 | 3 |
| `eval_foton_E7_olymp_04` | 4 | foton | а сейчас никак не можете сказать? просто хотела понять, есть ли см | draft_for_manager | 11.7 | 3 |
| `eval_foton_E7_olymp_05` | 2 | foton | а вы готовите к олимпиадам по физике? | draft_for_manager | 11.7 | 3 |
| `eval_foton_E8_exam_02` | 4 | foton | я поняла, но вы так и не сказали, это именно подготовка к ОГЭ или  | draft_for_manager | 11.7 | 3 |
| `eval_foton_E8_exam_03` | 4 | foton | я спрашивала не просто есть ли информатика, а ОГЭ отдельно от ЕГЭ  | draft_for_manager | 11.7 | 3 |
| `eval_foton_E8_exam_04` | 2 | foton | а к ЕГЭ по математике тоже готовите? | draft_for_manager | 11.7 | 3 |
| `eval_foton_E8_exam_05` | 2 | foton | а к ОГЭ по физике у вас есть подготовка для 10 класса? | draft_for_manager | 11.1 | 5 |
| `eval_foton_E9_trial_01` | 1 | foton | а это онлайн или очно, и по цене что выходит для 7 класса по матем | manager_only | 11.3 | 3 |
| `eval_foton_E9_trial_03` | 3 | foton | а на подготовительные по информатике онлайн можно записаться через | draft_for_manager | 11.7 | 3 |
| `eval_foton_E10_group_01` | 2 | foton | я же написала онлайн 🙂 а по дням групп подскажете? и сколько челов | draft_for_manager | 11.7 | 3 |
| `eval_foton_E10_group_03` | 3 | foton | а в группах ребята одного возраста обычно? | draft_for_manager | 11.7 | 3 |
| `eval_foton_C1_format_price_01` | 1 | foton | подскажите, а онлайн или очно можно? и по стоимости что выходит дл | manager_only | 11.3 | 3 |
| `eval_foton_C1_format_price_02` | 1 | foton | а онлайн или очно, и по стоимости что выходит для 9 класса? | manager_only | 11.3 | 3 |
| `eval_foton_C1_format_price_04` | 1 | foton | а онлайн или очно, и по стоимости что выходит для 6 класса? | draft_for_manager | 11.7 | 3 |
| `eval_foton_C1_format_price_05` | 1 | foton | а онлайн или очно, и по стоимости что выходит для 10 класса? | draft_for_manager | 11.7 | 3 |
| `eval_foton_C2_multi_subj_01` | 2 | foton | это за один предмет? а если математика и физика вместе, два предме | manager_only | 11.3 | 3 |
| `eval_foton_C2_multi_subj_01` | 3 | foton | я же спрашиваю за два предмета вместе, будет какая-то скидка или п | manager_only | 11.3 | 3 |
| `eval_foton_C2_multi_subj_01` | 4 | foton | а если три предмета взять, третий тоже со скидкой 30%? и сколько т | manager_only | 11.3 | 3 |
| `eval_foton_C2_multi_subj_01` | 5 | foton | понятно, вы так и не посчитали за 3 предмета, тогда пока не буду о | manager_only | 11.3 | 3 |
| `eval_foton_C2_multi_subj_02` | 1 | foton | если брать два предмета — физика и физика, 9 класс, очно — сколько | manager_only | 11.5 | 2 |
| `eval_foton_C2_multi_subj_02` | 2 | foton | а примерную цену сейчас можете сказать? мне надо понять, тяну ли д | manager_only | 11.3 | 3 |
| `eval_foton_C2_multi_subj_02` | 3 | foton | за год. то есть за два предмета сколько вместе получится? | manager_only | 11.3 | 3 |
| `eval_foton_C2_multi_subj_02` | 5 | foton | я спрашиваю именно итог за три предмета на год, можете просто посч | manager_only | 11.3 | 3 |
| `eval_foton_C2_multi_subj_03` | 1 | foton | здравствуйте, если 11 класс и брать онлайн информатику с физикой,  | manager_only | 11.3 | 3 |
| `eval_foton_C2_multi_subj_03` | 3 | foton | я про три предмета вместе спрашиваю, скидка на третий тоже будет и | manager_only | 11.3 | 3 |
| `eval_foton_C2_multi_subj_03` | 5 | foton | я же про два предмета спрашиваю, итоговую сумму можете посчитать?  | manager_only | 11.3 | 3 |
| `eval_foton_C2_multi_subj_04` | 2 | foton | это за один предмет или уже за математику с физикой вместе? | manager_only | 11.3 | 3 |
| `eval_foton_C2_multi_subj_04` | 3 | foton | а если три предмета взять, по цене как будет? | manager_only | 11.3 | 3 |
| `eval_foton_C2_multi_subj_04` | 4 | foton | я не поняла, итог за 3 предмета в год сколько будет? | draft_for_manager | 11.7 | 3 |
| `eval_foton_C2_multi_subj_04` | 5 | foton | ну ладно, тогда смысла нет, я хотела сразу понять порядок цены | draft_for_manager | 11.7 | 3 |
| `eval_foton_C2_multi_subj_05` | 2 | foton | а если три предмета брать, по цене как будет? | manager_only | 11.3 | 3 |
| `eval_foton_C2_multi_subj_05` | 3 | foton | то есть два предмета будут дешевле, да? можете итоговую сумму за г | manager_only | 11.3 | 3 |
| `eval_foton_C2_multi_subj_05` | 4 | foton | я поняла, но мне нужна именно сумма итогом за два предмета за год, | manager_only | 11.3 | 3 |
| `eval_foton_C2_multi_subj_05` | 5 | foton | ну вы так и не сказали итоговую сумму, мне так неудобно считать, у | manager_only | 11.3 | 3 |
| `eval_foton_C3_camp_refund_01` | 1 | foton | здравствуйте, подскажите ближайшую онлайн-смену по математике для  | manager_only | 11.3 | 3 |
| `eval_foton_C3_camp_refund_04` | 1 | foton | здравствуйте, подскажите ближайшую очную смену для 6 класса по мат | manager_only | 11.3 | 3 |
| `eval_foton_C3_camp_refund_05` | 1 | foton | здравствуйте, нужна физика онлайн для 10 класса на лето. какая бли | manager_only | 11.3 | 3 |
| `eval_foton_S1_camp_dates_02` | 1 | foton | какая ближайшая смена, где места есть по физике? 9 класс, очно | draft_for_manager | 11.7 | 3 |
| `eval_foton_S1_camp_dates_02` | 2 | foton | а примерно по датам можете сказать? вторая смена с конца июля, да? | draft_for_manager | 11.7 | 3 |
| `eval_foton_S1_camp_dates_02` | 3 | foton | понятно, просто мне сейчас даты нужны, без них не могу решить | draft_for_manager | 11.7 | 3 |
| `eval_foton_S1_camp_dates_03` | 1 | foton | здравствуйте, какая ближайшая смена есть по информатике? 11 класс, | draft_for_manager | 11.7 | 3 |
| `eval_foton_S1_camp_dates_04` | 1 | foton | здравствуйте, какая ближайшая смена есть по математике для 6 класс | draft_for_manager | 11.7 | 3 |
| `eval_foton_S1_camp_dates_04` | 2 | foton | а примерно по датам не подскажете? вторая смена с конца июля, да? | draft_for_manager | 11.7 | 3 |
| `eval_foton_S1_camp_dates_04` | 4 | foton | ясно, тогда подожду менеджера, потому что по датам так и не понятн | draft_for_manager | 11.7 | 3 |
| `eval_foton_S1_camp_dates_05` | 3 | foton | так а ближайшая смена когда? мне надо быстро понять, успеваем или  | draft_for_manager | 11.7 | 3 |
| `eval_foton_S3_vyezdnaya_02` | 3 | foton | а у вас выездных форматов больше нет? мне физика для 9 класса очно | draft_for_manager | 11.7 | 3 |
| `eval_foton_S3_vyezdnaya_02` | 4 | foton | ну мне сейчас понять надо, ждать Менделеево или искать другое мест | draft_for_manager | 11.1 | 5 |
| `eval_foton_S3_vyezdnaya_03` | 3 | foton | я про это и спрашиваю) у вас выездных форматов больше нет? | draft_for_manager | 11.1 | 5 |
| `eval_foton_S3_vyezdnaya_04` | 3 | foton | не очень поняла) у вас выездных форматов больше нет? | draft_for_manager | 11.7 | 3 |
| `eval_foton_S3_vyezdnaya_05` | 4 | foton | поняла. а у вас выездных форматов больше нет? именно по физике/лет | draft_for_manager | 11.1 | 5 |
| `eval_foton_S2_camp_price_02` | 1 | foton | здравствуйте, подскажите сколько стоит летняя смена по физике для  | draft_for_manager | 11.1 | 5 |
| `eval_foton_S2_camp_price_02` | 2 | foton | я же написала: физика, 9 класс, очно. сколько стоит смена? | manager_only | 11.3 | 3 |
| `eval_foton_S2_camp_price_03` | 2 | foton | а вы сами цену не видите? мне бы понять хотя бы порядок стоимости | manager_only | 11.3 | 3 |
| `eval_foton_S2_camp_price_03` | 4 | foton | поняла, но все же это смена или семестр? и как оплачивать, сразу в | manager_only | 11.3 | 3 |
| `eval_foton_S2_camp_price_04` | 1 | foton | здравствуйте, сколько стоит смена у вас? | draft_for_manager | 11.7 | 3 |
| `eval_foton_S2_camp_price_04` | 2 | foton | а хотя бы примерно можете сказать? и что входит в стоимость смены? | draft_for_manager | 11.7 | 3 |
| `eval_foton_S2_camp_price_04` | 3 | foton | ну вы прям ничего сами не знаете? тогда хотя бы скажите, оплата см | draft_for_manager | 11.7 | 3 |
| `eval_foton_S4_camp_who_01` | 1 | foton | а это онлайн или очно, и по цене что выходит для 7 класса? | manager_only | 11.3 | 3 |
| `eval_foton_S4_camp_who_02` | 1 | foton | подскажите, смена у вас для какого возраста? | draft_for_manager | 11.7 | 3 |
| `eval_foton_S4_camp_who_05` | 1 | foton | а это онлайн или очно, и по цене что выходит для 10 класса? | manager_only | 11.3 | 3 |
| `eval_foton_V4_platform_03` | 4 | foton | я поняла про пароль, но сам вход где искать? ссылку пришлют после  | draft_for_manager | 11.7 | 3 |
| `eval_foton_V5_tax_02` | 2 | foton | фио пока не готова писать, курс физика 9 класс очно. а налоговый в | manager_only | 11.3 | 3 |
| `eval_foton_V6_miss_02` | 1 | foton | а это онлайн или очно? и по цене что выходит для 9 класса по физик | manager_only | 11.3 | 3 |
| `eval_foton_V6_miss_03` | 2 | foton | а если ребенок заболеет в середине курса, больничные как у вас? | draft_for_manager | 11.7 | 3 |
| `eval_unpk_E1_price_03` | 4 | unpk | понятно, цены так и не увидела. тогда не буду пока оставлять заявк | draft_for_manager | 11.7 | 3 |
| `eval_unpk_E1_price_05` | 1 | unpk | здравствуйте, формат какой и сколько стоит информатика онлайн для  | draft_for_manager | 11.7 | 3 |
| `eval_unpk_E2_format_record_05` | 1 | unpk | здравствуйте, информатика для 10 класса онлайн есть? и сколько сто | draft_for_manager | 11.7 | 3 |
| `eval_unpk_E3_schedule_04` | 1 | unpk | здравствуйте, физика для 6 класса очно есть? и сколько стоит? | draft_for_manager | 11.7 | 3 |
| `eval_unpk_E3_schedule_05` | 1 | unpk | здравствуйте, это онлайн или очно? и по цене что выходит для 10 кл | draft_for_manager | 11.7 | 3 |
| `eval_unpk_E4_level_02` | 1 | unpk | здравствуйте, 9 класс, информатика очно. раньше почти не занималис | draft_for_manager | 11.7 | 3 |
| `eval_unpk_E4_level_02` | 3 | unpk | а если олимпиадную попробовать, туда с нуля берут или сначала тест | draft_for_manager | 11.7 | 3 |
| `eval_unpk_E4_level_04` | 2 | unpk | мне вообще очно нужно, не онлайн. и мы физикой раньше не занималис | draft_for_manager | 11.7 | 3 |
| `eval_unpk_E5_payment_01` | 1 | unpk | а оплата как проводится? | manager_only | 11.3 | 3 |
| `eval_unpk_E5_payment_02` | 1 | unpk | а оплата как проводится? | manager_only | 11.3 | 3 |
| `eval_unpk_E5_payment_03` | 1 | unpk | а оплата как проводится? | manager_only | 11.3 | 3 |
| `eval_unpk_E5_payment_05` | 2 | unpk | я же написала 10 класс, информатика онлайн. стоимость какая и опла | manager_only | 11.3 | 3 |
| `eval_unpk_E5_payment_05` | 4 | unpk | ну ок, тогда пусть менеджер напишет цену и как можно оплатить част | draft_for_manager | 11.7 | 3 |
| `eval_unpk_E6_discount_01` | 1 | unpk | а какие у вас скидки? | draft_for_manager | 11.7 | 3 |
| `eval_unpk_E6_discount_02` | 1 | unpk | а какие у вас скидки? | manager_only | 11.3 | 3 |
| `eval_unpk_E6_discount_02` | 2 | unpk | у нас в семье двое занимаются — есть скидка? | draft_for_manager | 11.7 | 3 |
| `eval_unpk_E6_discount_02` | 3 | unpk | а за второй предмет дешевле? мне просто быстро понять надо | draft_for_manager | 11.7 | 3 |
| `eval_unpk_E6_discount_02` | 4 | unpk | понятно, тогда не буду ждать, поищу на сайте сама | draft_for_manager | 11.7 | 3 |
| `eval_unpk_E6_discount_04` | 1 | unpk | здравствуйте, подскажите какой формат и сколько стоит физика для 6 | manager_only | 11.3 | 3 |
| `eval_unpk_E6_discount_04` | 2 | unpk | а какие у вас вообще скидки есть? | manager_only | 11.3 | 3 |
| `eval_unpk_E6_discount_05` | 1 | unpk | здравствуйте, информатика для 10 класса онлайн есть? и сколько сто | manager_only | Phase1 | после волн (вне P11) |
| `eval_unpk_E6_discount_05` | 2 | unpk | а цену так и не написали) и какие у вас скидки есть? | draft_for_manager | 11.7 | 3 |
| `eval_unpk_E6_discount_05` | 3 | unpk | мне бы хотя бы по скидкам понять сейчас, у нас в семье двое занима | draft_for_manager | 11.7 | 3 |
| `eval_unpk_E7_olymp_04` | 4 | unpk | а для 6 класса к олимпиадным задачам можно прокачаться или физтех  | manager_only | 11.3 | 3 |
| `eval_unpk_E7_olymp_05` | 3 | unpk | это именно физтех, да? а для 10 класса по информатике реально есть | draft_for_manager | 11.1 | 5 |
| `eval_unpk_E8_exam_01` | 2 | unpk | я же написала, 7 класс. физика онлайн интересует | draft_for_manager | 11.7 | 3 |
| `eval_unpk_E8_exam_02` | 2 | unpk | а к ЕГЭ по информатике готовите? | draft_for_manager | 11.1 | 5 |
| `eval_unpk_E8_exam_04` | 1 | unpk | а это онлайн или очно, и по цене что выходит для 6 класса? | draft_for_manager | 11.7 | 3 |
| `eval_unpk_E8_exam_05` | 4 | unpk | я же спрашиваю не про платформу, а огэ это отдельный курс или в 10 | draft_for_manager | 11.7 | 3 |
| `eval_unpk_E9_trial_01` | 4 | unpk | поняла, а пробное занятие есть перед записью? | draft_for_manager | 11.7 | 3 |
| `eval_unpk_E9_trial_04` | 3 | unpk | поняла, а с кем можно пообщаться по записи на пробное? | draft_for_manager | 11.7 | 3 |
| `eval_unpk_E10_group_03` | 1 | unpk | здравствуйте, подскажите сколько человек обычно в группе по матема | draft_for_manager | 11.7 | 3 |
| `eval_unpk_E10_group_04` | 1 | unpk | здравствуйте, какой формат занятий и сколько стоит физика для 6 кл | draft_for_manager | 11.7 | 3 |
| `eval_unpk_E10_group_04` | 3 | unpk | то есть по физике для 6 класса можно и индивидуально очно, да? или | draft_for_manager | 11.7 | 3 |
| `eval_unpk_C1_format_price_01` | 1 | unpk | здравствуйте, подскажите физика 7 класс онлайн есть? и сколько по  | draft_for_manager | 11.7 | 3 |
| `eval_unpk_C1_format_price_02` | 1 | unpk | а онлайн или очно, и по стоимости что выходит для 9 класса? | manager_only | 11.3 | 3 |
| `eval_unpk_C1_format_price_03` | 1 | unpk | здравствуйте, подскажите, математика 11 класс есть онлайн? и по ст | draft_for_manager | 11.7 | 3 |
| `eval_unpk_C1_format_price_04` | 1 | unpk | здравствуйте, а онлайн или очно физика для 6 класса? и по стоимост | manager_only | 11.3 | 3 |
| `eval_unpk_C1_format_price_05` | 3 | unpk | понятно, тогда не надо. мне нужна была цена и формат сразу, без ме | draft_for_manager | 11.7 | 3 |
| `eval_unpk_C2_multi_subj_01` | 1 | unpk | если брать два предмета, физику и физику, 7 класс онлайн — сколько | manager_only | 11.3 | 3 |
| `eval_unpk_C2_multi_subj_01` | 2 | unpk | ну не две физики конечно, я про физику плюс еще один предмет. а ес | manager_only | 11.3 | 3 |
| `eval_unpk_C2_multi_subj_02` | 1 | unpk | здравствуйте, если брать два предмета — информатика и физика, 9 кл | manager_only | 11.3 | 3 |
| `eval_unpk_C2_multi_subj_02` | 2 | unpk | так а итогом за информатику и физику сколько будет за год? | manager_only | 11.3 | 3 |
| `eval_unpk_C2_multi_subj_02` | 3 | unpk | ну посчитайте пожалуйста итог, я тороплюсь) 82 плюс второй со скид | manager_only | 11.3 | 3 |
| `eval_unpk_C2_multi_subj_02` | 4 | unpk | то есть 147 600 за два предмета за год? правильно понимаю? | manager_only | 11.3 | 3 |
| `eval_unpk_C2_multi_subj_02` | 5 | unpk | ну блин, мне просто сумму надо было подтвердить, ладно тогда | manager_only | 11.3 | 3 |
| `eval_unpk_C2_multi_subj_03` | 1 | unpk | здравствуйте, если брать два предмета — математику и физику, 11 кл | manager_only | 11.3 | 3 |
| `eval_unpk_C2_multi_subj_03` | 2 | unpk | а сумма-то какая будет? и если три предмета брать, по цене как? | manager_only | 11.3 | 3 |
| `eval_unpk_C2_multi_subj_03` | 3 | unpk | ну я пока не решила третий, мне надо понять порядок цены. два пред | manager_only | 11.3 | 3 |
| `eval_unpk_C2_multi_subj_04` | 3 | unpk | а если три предмета брать, по цене как будет? | manager_only | 11.3 | 3 |
| `eval_unpk_C2_multi_subj_04` | 4 | unpk | то есть два предмета на год будет 82 000 плюс второй со скидкой 20 | manager_only | 11.3 | 3 |
| `eval_unpk_C2_multi_subj_04` | 5 | unpk | ну я же спрашиваю итог за два предмета, вы сумму не считаете. ладн | manager_only | 11.3 | 3 |
| `eval_unpk_C2_multi_subj_05` | 1 | unpk | здравствуйте, если брать два предмета — информатику и физику, 10 к | manager_only | 11.3 | 3 |
| `eval_unpk_C2_multi_subj_05` | 3 | unpk | понятно, а если три предмета взять, там по цене как считается? | draft_for_manager | 11.7 | 3 |
| `eval_unpk_C2_multi_subj_05` | 4 | unpk | то есть даже порядок цены не скажете? два предмета хоть дешевле вы | manager_only | 11.3 | 3 |
| `eval_unpk_C2_multi_subj_05` | 5 | unpk | ладно, тогда смысла нет, мне нужна была хотя бы сумма, а не только | manager_only | 11.3 | 3 |
| `eval_unpk_C3_camp_refund_02` | 1 | unpk | здравствуйте, ближайшая очная смена по информатике для 9 класса ко | draft_for_manager | 11.7 | 3 |
| `eval_unpk_C3_camp_refund_03` | 1 | unpk | здравствуйте, подскажите какая ближайшая онлайн-смена по математик | draft_for_manager | 11.7 | 3 |
| `eval_unpk_C3_camp_refund_04` | 1 | unpk | здравствуйте, подскажите ближайшую очную смену по физике для 6 кла | draft_for_manager | 11.7 | 3 |
| `eval_unpk_S1_camp_dates_01` | 1 | unpk | какая ближайшая смена, где места есть по физике? | draft_for_manager | 11.7 | 3 |
| `eval_unpk_S1_camp_dates_02` | 1 | unpk | подскажите, какая ближайшая смена есть очно по информатике для 9 к | draft_for_manager | 11.7 | 3 |
| `eval_unpk_S1_camp_dates_02` | 3 | unpk | а вторая смена с конца июля начинается, да? | draft_for_manager | 11.7 | 3 |
| `eval_unpk_S1_camp_dates_03` | 2 | unpk | а летняя смена есть? я думала вторая смена с конца июля, нет? | draft_for_manager | 11.7 | 3 |
| `eval_unpk_S1_camp_dates_04` | 2 | unpk | очно. а какая ближайшая смена, где места есть по физике? | draft_for_manager | 11.7 | 3 |
| `eval_unpk_S1_camp_dates_04` | 3 | unpk | а примерно по датам можете сказать? вторая смена с конца июля, да? | draft_for_manager | 11.7 | 3 |
| `eval_unpk_S1_camp_dates_05` | 3 | unpk | то есть дат вообще нет? даже примерно, вторая смена с конца июля н | draft_for_manager | 11.7 | 3 |
| `eval_unpk_S3_vyezdnaya_01` | 4 | unpk | я про выездные именно, кроме Менделеево есть что-то? | draft_for_manager | 11.1 | 5 |
| `eval_unpk_S3_vyezdnaya_03` | 3 | unpk | у вас выездных форматов больше нет? просто ищу именно что-то по ма | draft_for_manager | 11.7 | 3 |
| `eval_unpk_S3_vyezdnaya_05` | 1 | unpk | по выездной школе в Менделеево можно вопрос? это онлайн или очно,  | draft_for_manager | 11.7 | 3 |
| `eval_unpk_S3_vyezdnaya_05` | 3 | unpk | я же спрашивала по цене и формату для 10 класса, но ладно. у вас в | draft_for_manager | 11.1 | 5 |
| `eval_unpk_S2_camp_price_01` | 3 | unpk | я же выше написала, 7 класс) а оплата смены как проводится? | manager_only | 11.3 | 3 |
| `eval_unpk_S2_camp_price_02` | 1 | unpk | здравствуйте, подскажите сколько стоит смена на лето? интересует и | manager_only | 11.3 | 3 |
| `eval_unpk_S2_camp_price_02` | 2 | unpk | а что входит в эту стоимость? | draft_for_manager | 11.7 | 3 |
| `eval_unpk_S2_camp_price_02` | 3 | unpk | ну а хотя бы примерно что туда входит? проживание, питание есть? и | draft_for_manager | 11.7 | 3 |
| `eval_unpk_S2_camp_price_03` | 1 | unpk | здравствуйте, сколько стоит смена на лето? интересует математика 1 | manager_only | 11.3 | 3 |
| `eval_unpk_S2_camp_price_04` | 2 | unpk | мне нужно очно, не онлайн. и цену за смену подскажите для 6 класса | draft_for_manager | 11.7 | 3 |
| `eval_unpk_S2_camp_price_04` | 3 | unpk | а что вообще входит в стоимость смены? просто цену тоже хотелось б | manager_only | 11.3 | 3 |
| `eval_unpk_V2_are_you_bot_01` | 2 | unpk | а цену так и не написали) и это бот или живой человек? | draft_for_manager | 11.7 | 3 |
| `eval_unpk_V5_tax_03` | 1 | unpk | здравствуйте, подскажите, у вас подготовка по математике для 11 кл | manager_only | 11.3 | 3 |

## Открытые вопросы

1. Классификация — машинная по приоритету флагов; редакторы (11.9, 83 со-присутствий) часто поглощены классом 11.7/11.3 как вторичные — при желании можно дать мульти-метку.
2. 11.7 (88) — крупнейший класс; часть может быть «факт не использован» (Phase 1), маскированной verifier-fallback'ом. Точное разделение требует `answerability` контракта по каждому ходу.
3. Метрика `false_handoff` не отражает Волну 1 (P0). Для оценки эффекта Волны 1 нужна метрика `over_handoff_turn_rate`/доля `manager_only` без факта — отдельно.
