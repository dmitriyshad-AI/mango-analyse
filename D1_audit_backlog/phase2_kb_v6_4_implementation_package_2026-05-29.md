# Phase 2 — пакет данных KB v6.4 (реализация)

Автор: второй Claude. Дата: 2026-05-29. Снимок: HEAD `36e23cb8`. Read-only.

## ВАЖНО — путь внесения (по CLAUDE.md)

KB собирается ТОЛЬКО через `scripts/build_kb_release_v6_1_team_answers.py` (применяет `release_manifest.yaml` из `_sources/`). **Новые факты вносятся в source-YAML (`_sources/facts/facts_for_bot_FOTON.yaml` / `..._UNPK.yaml`), НЕ правкой `client_safe_facts_*.jsonl` напрямую.** Сборщик сам генерирует `fact_id`, `source_sha256`, `short_fact` и т.п. Ниже даю клиентски-значимые поля; builder-managed поля (`fact_id/source_sha256/source_*`) — авто. Схема записи — по образцу `presentation_format_facts_2026_05_21.client_safe_facts.refund_presale_policy.client_safe_text`.

**Схема `applies_to` (решение для Кодекса):** в v6.3 поля `applies_to.*` на уровне записи НЕТ; предлагаю класть в `structured_value.applies_to = {grades:[...], subjects:[...], formats:[...], frequency:"...", products:[...]}` + добавить `structured_value.is_positive_statement: bool`. Это не ломает схему (structured_value — свободный dict). Подтвердить.

---

## 1. JSON-фрагменты новых фактов

### 1.1. `refund_post_payment.client_safe_text` — ОБА бренда
Формулировка-кандидат из `TZ_wave_1a_FINAL §5.1` (Claude #1). **ТРЕБУЕТ ПОДТВЕРЖДЕНИЯ ДМИТРИЯ** (это новый клиентский текст про возврат ПОСЛЕ оплаты — soft handoff Спор 1).

Foton (`client_safe_facts_foton.jsonl` / source FOTON.yaml):
```json
{
  "fact_key": "presentation_format_facts_2026_05_21.client_safe_facts.refund_post_payment.client_safe_text",
  "brand": "foton", "active_brand_scope": "foton_bot",
  "allowed_for_client_answer": true, "fact_type": "policy", "fact_types": ["policy"],
  "client_safe_text": "По возврату средств после оплаты нужен расчёт менеджера: он посмотрит, какая часть курса уже пройдена, и пришлёт точную сумму к возврату. Я уже передал ему ваш запрос — он свяжется в рабочее время.",
  "cross_brand_policy": "active_brand_only",
  "forbidden_client_mentions": ["УНПК","АНО ДПО","НОУ УНПК","kmipt.ru","@unpk_mipt"],
  "forbidden_promises": ["Не называть конкретную сумму возврата без расчёта менеджера."],
  "risk_level": "high", "route_policy": "draft_for_manager",
  "requires_manager_confirmation": true, "usable_for_precise_answer": false,
  "structured_value": {"path":"...refund_post_payment.client_safe_text","valid_until":"2026-12-31",
     "applies_to":{"refund_stage":"post_payment"}, "is_positive_statement": true},
  "valid_until": "2026-12-31", "verification_status": "verified",
  "notes": "Phase 12 TZ-soft-handoff (Спор 1). ТРЕБУЕТ подтверждения формулировки Дмитрием."
}
```
УНПК: тот же объект с `brand:"unpk"`, `active_brand_scope:"unpk_bot"`, `forbidden_client_mentions:["Фотон","ЦДПО","Долями","Т-Банк","Скорняжный","cdpofoton.ru"]`, тот же текст без бренд-имени.

### 1.2. Онлайн-цена УНПК (Б.3 — частота «2 раза в неделю»)
**КОНТЕКСТ:** число 41 800/69 900 лежит в `manager_only` с пометкой «будничные онлайн-курсы прошлого года» → **сумму НЕ выдумываю**, ставлю плейсхолдер. Частота «2 раза в неделю по 90 минут» подтверждена фактом `online_courses_format`.
```json
{
  "fact_key": "prices_regular_2026_27.online_5_11_class_regular.client_safe_text",
  "brand": "unpk", "active_brand_scope": "unpk_bot", "allowed_for_client_answer": true,
  "fact_type": "price", "fact_types": ["price"],
  "client_safe_text": "УНПК: онлайн-курсы для 5–11 классов — занятия 2 раза в неделю по 90 минут на МТС-Link. Стоимость на 2026/27: <ТРЕБУЕТ ПОДТВЕРЖДЕНИЯ ДМИТРИЯ: актуальная сумма семестр/год>.",
  "route_policy": "draft_for_manager", "usable_for_precise_answer": false,
  "structured_value": {"format":"online","path":"...online_5_11_class_regular.client_safe_text",
     "applies_to":{"grades":[5,6,7,8,9,10,11],"formats":["online"],"frequency":"2 раза в неделю по 90 минут"},
     "amount_semester": null, "amount_year": null, "is_positive_statement": true, "valid_until":"2027-08-31"},
  "notes": "Phase 2/TZ-05. Сумму внести только после подтверждения актуальности (стар. 41800/69900 помечены как прошлогодние в manager_only)."
}
```
До подтверждения суммы — `usable_for_precise_answer:false` (бот честно «цену уточнит менеджер»).

### 1.3. Отрицательный факт closed-world (S3) — `is_positive_statement: false`
**ТРЕБУЕТ ПОДТВЕРЖДЕНИЯ ДМИТРИЯ** (закрытое утверждение).
```json
{
  "fact_key": "lvsh_mendeleevo_2026.vyezdnaya_no_other_formats_unpk",
  "brand": "unpk", "active_brand_scope": "unpk_bot", "allowed_for_client_answer": true,
  "fact_type": "program", "fact_types": ["program"],
  "client_safe_text": "У УНПК выездной формат — это выездная школа ЛВШ Менделеево; других выездных форматов нет.",
  "route_policy": "bot_answer_self_for_pilot", "usable_for_precise_answer": true,
  "structured_value": {"path":"...vyezdnaya_no_other_formats_unpk","is_positive_statement": false,
     "applies_to":{"products":["residential_lvsh"]}, "valid_until":"2027-08-31"},
  "notes": "Phase 2/TZ-09 closed-world. ТРЕБУЕТ подтверждения Дмитрием (закрытое утверждение)."
}
```
(Аналогично — кандидаты на отрицательные факты: «банковской рассрочки у УНПК нет» — частично есть; «ОГЭ-интенсива у Фотона нет/есть» — уточнить. Заводить по мере подтверждения.)

### 1.4. `applies_to.grades: [9, 11]` — на каких ключах
Добавить `structured_value.applies_to.grades=[9,11]` (+ `subjects`, `formats`) на олимпиадные Физтех-факты УНПК:
- `prices_regular_2026_27.online_olympiad_phystech_classes.client_safe_text` — grades [9,11], formats ["online"]. (уже несёт «9 и 11» в тексте — формализовать в structured.)
- `prices_regular_2026_27.online_olympiad_phystech_classes.product` — то же.
- `tg_unpk_verified_2026_05_21.client_facts.olympiad_targets.client_safe_text` — **ОТКРЫТО Дмитрию:** олимпиадная подготовка в целом (вкл. очное) ограничена 9/11 или только онлайн-продукт? От этого зависит grades для очного (TZ-08). Если только онлайн — на `olympiad_targets` grades НЕ ставить, а завести отдельный очный олимпиадный факт с подтверждённым набором.
- manager_only `...online_olympiad_phystech_9_and_11.note_internal` — это и есть якорь набора [9,11] (но устарел по цене).

---

## 2. Унификация 8 SAFE пар (из `kb_v64_unification_safety_check`)

Только SAFE (значения идентичны, отличие лишь в имени/индексе/брендовом суффиксе). BLOCKED-пары НЕ трогать.

| # | действие | foton-ключ | unpk-ключ | примечание |
|---|---|---|---|---|
| 1 | оставить как есть (уже бренд-суффикс) | `brand_rules.approved_brand_relationship_answer.foton` | `...unpk` | унификация не нужна |
| 2 | свести к одному имени | `discounts.stacking_rule` | `discounts.stacking_rule_text` | выбрать канон `stacking_rule_text`, удалить дубль |
| 3 | то же (зеркало #2) | `discounts.stacking_rule_text` | `discounts.stacking_rule` | устранить внутрибрендовый дубль `stacking_rule`/`_text` |
| 4 | оставить (уже бренд-суффикс) | `bot_policy.approved_phrases.theme_11_contract.foton` | `...unpk` | — |
| 5 | оставить (уже бренд-суффикс) | `bot_policy.approved_phrases.theme_12_certificate.foton` | `...unpk` | — |
| 6 | оставить (уже бренд-суффикс) | `bot_policy.approved_phrases.theme_17_teachers.foton` | `...unpk` | — |
| 7 | унифицировать индекс | `objection_responses.too_expensive_course.3` | `objection_responses.too_expensive_course.5` | один контент (кэшбек 10 000), разный индекс → единая индексация |
| 8 | свести namespace | `objection_responses.brand_link_question.approved_response` | `brand_rules.approved_brand_relationship_answer.unpk` | один ответ о связи брендов в разных namespace → канон `brand_rules.approved_brand_relationship_answer` |

Главное действие: устранить **внутрибрендовый дубль** `discounts.stacking_rule` vs `stacking_rule_text` (источник `other_brand_match`-шума судьи) и свести brand-link в один namespace.

---

## 3. Перенос 8 фактов manager_only → client-safe

Из 35 кандидатов (`kb_v64_manager_only_to_client_safe`). Готовый client-safe текст:

| manager_only_fact_key | бренд | should_move | client-safe текст (предложение) |
|---|---|---|---|
| `installment.client_confirmed_terms.client_safe_text` | foton | да | «В Фотоне оплата частями: 6/10/12 месяцев + Долями…» (route уже `bot_answer_self_for_pilot`) |
| `certificates.client_safe_text.when_asked` | foton | да | «Менеджер подготовит справку и пришлёт в течение 10 рабочих дней…» |
| `prices_regular_2026_27.online_5_11_class_regul...` | unpk | да (см. п.1.2) | онлайн-цена 5-11 — после подтверждения суммы |
| `fiztech_olympiad.prices.group` | unpk | условно | «олимпиадная подготовка, группа — 50 000 ₽» — **подтвердить актуальность** |
| `fiztech_olympiad.prices.individual` | unpk | условно | «индивидуально — 33 000 ₽» — подтвердить |
| `team_answers.q15.unpk_online_other_classes...` | unpk | да | «по онлайн вне олимпиадной Физтех 9/11 — уточнит менеджер» |
| `team_answers.q11.modular_courses_m9_m11.old_pr*` | foton | **НЕТ** | М9/М11 discontinued (CLAUDE.md) — НЕ переносить |
| `intensives_2026.*` (ОГЭ/ЕГЭ состав) | оба | да | состав интенсива (психподдержка и т.п.) — client-safe |

⚠️ Все цены/устаревшие (`do_not_use_as_current_price`) — переносить ТОЛЬКО после подтверждения Дмитрием актуальности.

---

## 4. Таблица 250 фактов на ручной audit

Полная машинная таблица по ВСЕМ 838 фактам (с колонками applies_to/needs_audit/proposed) уже сформирована: **`kb_v64_enrichment_table_full.csv`** (фильтр `needs_manual_audit=true` → 250 строк). Здесь — формат + критичная подвыборка; полный список брать из CSV.

Распределение 250 по причинам: числовой факт (риск расхождения брендов) — 130; экзамен/интенсив — ~57; лагерь/смена с датой — ~28; уровни/разряды — ~24; олимпиадная подготовка — ~20.

Формат (как в задаче):
| fact_key | бренд | текущий applies_to.* | предложение applies_to.* | требует Дмитрия |
|---|---|---|---|---|
| `prices_regular_2026_27.online_olympiad_phystech_classes.client_safe_text` | unpk | — | grades [9,11], formats[online] | **да** (очное?) |
| `tg_unpk...olympiad_targets.client_safe_text` | unpk | — | grades ? | **да** (вкл. очное?) |
| `intensives_2026.oge_foton.classes` | foton | 8-9 (в тексте) | grades [8,9], products[exam_track_oge] | да (ОГЭ=9) |
| `intensives_2026.ege_intensive.*` | unpk | — | grades [11], products[exam_track_ege] | да (ЕГЭ=11) |
| `online_platform.levels.1/2/3` | оба | — | levels[базовый/продвинутый/олимпиадный] | нет (механически) |
| `ls_city_2026_*.smeny/dates/*` | оба | — | products[city_camp], dates | нет (из текста) |
| `lvsh_mendeleevo_2026.*` | оба | — | products[residential_lvsh], dates | частично (даты — да) |
| `discounts.second_subject.*.pct` | оба | format в structured | formats[offline/online] | нет |
| `prices_regular_2026_27.offline_5_11_class.*` | оба | classes "5-11" | grades [5..11], formats[offline] | нет (механически) |
| (… 240 строк — в `kb_v64_enrichment_table_full.csv`) | | | | |

Правило «требует Дмитрия = да»: олимпиада/экзамен (набор классов спорен, риск расхождения брендов) и любые числовые цены/скидки, потенциально различающиеся между брендами. «нет» — механический вывод из `structured_value.format`/`classes`.

## Открытые вопросы Дмитрию/Кодексу
1. Формулировка `refund_post_payment` (п.1.1) — подтвердить текст (Спор 1).
2. Актуальная онлайн-цена УНПК 5-11 (п.1.2) — старые 41800/69900 помечены прошлогодними; дать актуальные или оставить «уточнит менеджер».
3. Очная олимпиадная Физтех-подготовка ограничена 9/11 (как онлайн) или иначе (п.1.4, критично для TZ-08).
4. Текст отрицательного факта «других выездных нет» (п.1.3, TZ-09) — подтвердить закрытое утверждение.
5. Схема `applies_to`/`is_positive_statement`: в `structured_value` (предложено) или отдельным полем записи — решение Кодекса/сборщика v6.1.
6. Цены fiztech_olympiad / интенсивов из manager_only (п.3) — актуальны для переноса в client-safe?
