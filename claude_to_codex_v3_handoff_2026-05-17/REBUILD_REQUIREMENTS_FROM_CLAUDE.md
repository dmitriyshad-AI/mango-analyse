# Требования к повторной развёртке facts_registry (v3)
Дата: 2026-05-17 (вечер)
От: Claude
Кому: Codex
Контекст: ревью твоего kb_release_20260517_v2_handoff_for_claude_and_team выявил критические баги контентного слоя при сохранении правильной архитектуры.

---

## TL;DR трёх главных правил

1. **Каждое число — отдельный fact_id** с заполненным fact_text. Не сжимай словари в одну запись.
2. **`forbidden_to_say` — это пост-фильтр, не контент**. Никогда не разворачивай содержимое этих полей как client_safe_text.
3. **`internal_only_for_number: true` означает «не показывать клиенту»**. Номер лицензии не должен попадать в client_safe_text.

---

## БАГ 1. Сжатие словарей в одну запись

### Что произошло в v2

В моих YAML есть сложные блоки, например:

```yaml
prices_regular_2026_27:
  status: verified
  brand: foton
  offline_5_11_class:
    before_2026_07_01: { semester: 44600, year: 74500 }
    after_2026_07_01:
      four_weeks: 11400
      semester_range: [37125, 57000]
      year_range: [83500, 99750]
```

В facts_registry v2 этот блок попал как **одна запись** с пустым `fact_text` и потерянным числовым содержимым. В результате в реестре нет ни одного из чисел 44 600 / 74 500 / 11 400 / 37 125 / 57 000 / 83 500 / 99 750.

То же случилось с блоками:
- `discounts` (Фотон и УНПК)
- `lvsh_mendeleevo_2026`
- `individual_lessons_foton`
- `modular_courses_m9_m11`
- `zvsh_mendeleevo`
- `fiztech_olympiad`
- `preschool_patsayeva`
- `intensives_2026`

### Что нужно сделать в v3

Развернуть каждое число / условие как отдельный fact с заполненными полями. Пример правильной развёртки для одного блока цен:

```yaml
- fact_id: "foton:price:offline_5_11:before_2026_07_01:semester"
  fact_key: "prices_regular.foton.offline.5_11.before_2026_07_01.semester"
  fact_type: "price"
  brand: "foton"
  product: "regular_courses_offline_5_11"
  fact_text: "Семестр очно 5-11 класс до 01.07.2026: 44 600 ₽"
  client_safe_text: "Семестр обучения очно для 5-11 классов (при оплате до 1 июля) — 44 600 ₽"
  manager_check_text: "Подтвердить, что прайс не изменился с 16.03.2026"
  structured_value:
    amount: 44600
    currency: "RUB"
    period: "semester"
    classes: "5-11"
    format: "offline"
    valid_until: "2026-07-01"
  source_id: "gdrive:foton_prices_2026_2027"
  source_sha256: "<sha256 из source_inventory>"
  freshness_status: "document_verified"
  allowed_for_client_answer: true
  active_brand_scope: "foton_bot"
  route_policy: "bot_answer_self_for_pilot"
```

Аналогично — для каждого числа в каждом из 8 сложных блоков. Получится ~120-150 числовых фактов вместо 10 пустых.

### Контрольная проверка

После развёртки в facts_registry должны находиться grep'ом следующие числа:
- 44600, 74500, 49000, 82000, 29750, 47250 (цены курсов 2026/27)
- 98000, 75000, 120000, 89900, 83800 (цены ЛВШ)
- 16900, 27720, 18800, 34400 (интенсивы)
- 3900, 6900, 23000 (индивидуальные)
- 18900, 94500 (модульные)
- 33000, 50000 (ФизТех)
- 11900, 56500, 94000 (дошкольники Пацаева)

Если хоть одного числа нет — баг 1 не исправлен.

---

## БАГ 2. Cross-brand утечка через forbidden_to_say

### Что произошло в v2

В моих YAML каждый блок client_safe_text содержит подсписок:

```yaml
client_safe_text:
  when_asked: "У нас есть рассрочка через Т-Банк..."
  forbidden_to_say:
    - "в УНПК рассрочки нет"
    - "у наших партнёров"
    - "Фотон даёт, а другие нет"
```

В facts_registry v2 элементы списка `forbidden_to_say` развёрнуты как отдельные факты с `client_safe_text="в УНПК рассрочки нет"`, `brand=foton`, `allowed_for_client_answer=true`. То есть мои запреты превратились в разрешённый клиентский текст. Это самая опасная утечка.

### Что нужно сделать в v3

Поле `forbidden_to_say` — это **пост-фильтр**, не контент. При парсинге YAML:

1. Содержимое `forbidden_to_say` НЕ создаёт записей в facts_registry.
2. Содержимое объединяется с глобальным списком из `brand_rules.yaml::forbidden_client_mentions` и применяется на этапе пост-фильтра draft_text.
3. Если какая-то фраза из draft_text совпадает с любым элементом любого `forbidden_to_say` — draft помечается `brand_separation_violation` и идёт в `manager_only`.

### Контрольная проверка

В facts_registry v3 не должно быть ни одной записи, где:
- `client_safe_text` начинается с «в УНПК», «в Фотоне», «у наших партнёров»
- `client_safe_text` содержит названия двух брендов одновременно
- `brand=foton` и в client_safe_text есть слово «УНПК» (или наоборот)

Это `grep`-проверка: 0 совпадений — баг 2 исправлен.

---

## БАГ 3. Нарушение internal_only_for_number

### Что произошло в v2

В моих YAML по блокам лицензий:

```yaml
ano_dpo_unpk_mfti:
  status: verified
  number: "Л035-01255-50/01195871"
  date: "13.05.2024"
  holder: "АНО ДПО «УНПК МФТИ»"
  internal_only_for_number: true   # ВАЖНО

client_safe_summary: "У нас есть лицензия на образовательную деятельность."
```

В facts_registry v2 поля `number` и `date` развёрнуты как факты с `client_safe_text="Л035-01255-50/01195871 от 13.05.2024"`. То есть клиент бота УНПК увидит конкретный номер лицензии вопреки моему запрету.

### Что нужно сделать в v3

При парсинге блока с `internal_only_for_number: true`:

1. Поля `number`, `date`, `holder` НЕ попадают в `client_safe_text`.
2. Они могут быть в `manager_check_text` или `internal_text` — только для менеджера.
3. `client_safe_text` берётся ТОЛЬКО из `client_safe_summary` (если есть) или генерируется как обобщённая фраза «У нас есть лицензия на образовательную деятельность».

То же правило применить к:
- `responsible_person_internal` (Дарья Клычева, Кузнецова А.Е., Харламов М.Ю.)
- `legal_entity_internal` (АНО ДПО, ООО ЦДПО Фотон и т.д.)
- `license_basis_internal`

Любое поле с суффиксом `_internal` — manager_check_text, не client_safe_text.

### Контрольная проверка

`grep "Л035-01255" facts_registry.yaml` — 0 совпадений в полях с `allowed_for_client_answer=true`.
`grep "АНО ДПО" facts_registry.yaml` — допустимо только в `manager_check_text` или `brand_internal_note`, не в `client_safe_text`.

---

## БАГ 4. Разорванный JOIN source_id ↔ source_registry

### Что произошло в v2

Из аудита: 198 из 204 `source_id` в facts отсутствуют в source_registry. Codex использовал короткие ключи (`source:claude_layer:tutoring_individual`), которых нет в source_registry. JOIN не работает на 50% фактов.

### Что нужно сделать в v3

Каждый fact должен иметь `source_id`, который существует в source_registry. Если факт идёт из моих claude_layer YAML, привязать его к одному из 217 источников source_registry (или к новому source с явной записью в registry).

Если первичный источник факта — мой YAML, создать в source_registry запись:
```json
{
  "source_id": "claude_layer:facts_for_bot_FOTON",
  "source_kind": "claude_yaml",
  "title": "facts_for_bot_FOTON.yaml (Claude layer v3)",
  "path": "claude_to_codex_v3_handoff_2026-05-17/facts_for_bot_FOTON.yaml",
  "sha256": "<хеш файла>",
  "brand": "foton",
  "freshness_status": "fresh_verified"
}
```

И ссылаться на неё через `source_id="claude_layer:facts_for_bot_FOTON"` из всех порождённых фактов.

### Контрольная проверка

`SELECT COUNT(*) FROM facts WHERE source_id NOT IN (SELECT source_id FROM sources)` должен дать 0.

---

## ТРЕБОВАНИЕ 5. Учёт моих финальных правок от 17.05.2026

Codex v2 смотрел старую папку `Mango_Bot_Knowledge_Base_FINAL_2026-05-17`. В этой папке (`claude_to_codex_v3_handoff_2026-05-17`) — актуальное состояние со всеми обновлениями:

Что должно быть отражено в новом facts_registry:

1. **Долями Плюс Т-Банк** 3/6/10 мес, 0-16,9% (раньше — Тинькофф 6/12 мес). Закрывает q9.
2. **ЗВШ Менделеево теперь на АНО УНПК** (раньше на Фотоне). Бот Фотон не предлагает ЗВШ. Бот УНПК — да.
3. **Подписанты 2026**: Кузнецова А.Е. (Фотон, доверенность №6 от 17.11.2025), Харламов М.Ю. (АНО УНПК, доверенность №8 от 15.08.2025). Internal.
4. **Шкала удержаний при возврате**: Фотон 15 000 ₽ депозит + шкала 40/50/100 тыс ₽; АНО — 20/30/50/100 тыс ₽. Internal, не client_safe_text.
5. **Преподаватели Менделеево**: Кальцын В.А., Вагин К.Ю., Зиманова А.Л., Мендель А.В., Ибрагимов Р.Р., Белых К.И. — у УНПК (см. facts_for_bot_UNPK.yaml).
6. **Преподаватели интенсивов**: Редичкина и Петрунина (ОГЭ Фотон); Белых и Маилян (ЕГЭ УНПК).
7. **q14**: УНПК очно 49 000/82 000 — расхождение источников между «1-4 класс» и «5-11». Открытый вопрос.
8. **q15**: УНПК онлайн 41 800/69 900 — это олимпиадная подготовка Физтех, а не обычный курс. Помечено `status: needs_owner_confirmation` с полями `semester_provisional`, `year_provisional`.
9. **Лицензии**: 4 реальных номера, не цифра «70369» (она устарела).
   - АНО ДПО УНПК МФТИ — Л035-01255-50/01195871 от 13.05.2024
   - НОУ УНПК МФТИ — 50Л01 №0000547 от 06.03.2013
   - ООО ЦДПО Фотон — №77753 от 23.11.2018
   - ООО ЦРДО Фотон — Л035-01255-50/02496431 от 20.06.2025
10. **Маткапитал и налоговый вычет** — для обоих брендов (закрывает q1, q2).
11. **Новые продукты** Фотон: индивидуальные занятия (3 900 / 6 900 / 23 000), модульные М9/М11 (18 900-94 500).
12. **Новые продукты** УНПК: ФизТех-олимпиада (33 000 / 50 000), дошкольники Пацаева (11 900 / 56 500 / 94 000).
13. **Скидки**: refer_a_friend Фотон онлайн = 10 000 ₽ (не 5 000); active_student_to_summer_camp 7%; loyal_customers_camps 5/10/15/20/30%.
14. **НДС-льгота**: подп. 4 п. 2 ст. 149 НК РФ. Internal.
15. **Промокоды преподавателей**: ABRAMOV, VAGIN, DIGUROV, MAREEV (5%); Флоктори -10 000/-20 000 ₽. Internal до уточнения у маркетинга.
16. **Медсправки**: форма 079У валидна 3 мес, форма №291 — 3 дня.

---

## ТРЕБОВАНИЕ 6. approval_queue v3

В approval_queue v2 было 133 пункта, только 4 темы из 33 проработаны (маткап, налог, справки, юр.лица). РОПу нечего утверждать по существу бизнеса.

Что должно быть в approval_queue v3:
- Все цены 2026/27 поштучно (Фотон + УНПК, очно + онлайн, по классам, до/после дедлайнов)
- Все скидки поштучно (2-й предмет, многодетные, МФТИ, refer_a_friend, любимые клиенты, действующие, ранняя бронь)
- Все промокоды (LVSH-VEB20, LVSH-KF-10, преподавательские ABRAMOV/VAGIN/DIGUROV/MAREEV, Флоктори)
- Дедлайны (1 мая запись курсов, 15 мая оплата курсов, 1 июня ранняя бронь ЛВШ)
- ЛВШ цены и смены (отдельно Фотон и УНПК, по 4 смены каждый)
- Параметры программ (35 занятий, 20+15 недель, 6-12 чел в группе, и т.д.)
- Контакты (фразы которые бот может говорить про связь, не показывая внутренние)
- Параметры интенсивов (преподаватели, длительность, количество вебинаров)
- Условия рассрочки (Долями Плюс лимит, сроки, проценты — только Фотон)
- Лимиты налогового вычета 2024/2023

Структура каждого пункта:
```csv
priority,approval_item_id,item_type,topic,fact_id_ref,manager_text,suggested_decision,rop_question
P0,approve:foton:price:offline_5_11:semester:44600,price,01_pricing,foton:price:offline_5_11:before_2026_07_01:semester,"Семестр очно 5-11 до 01.07.2026: 44 600 ₽","fresh_verified — можно использовать в клиентском ответе","Подтверждаете эту цену для бота как точный факт?"
```

Цель: ~400-600 пунктов approval_queue, сгруппированных по 33 темам моего опросника, где каждый пункт = один факт с конкретным числом или условием. РОП проходит за 3-4 часа, отмечая ☑/☐.

См. источник для структуры: `enrichment_log/07_approval_queue_grouped.md` в исходной папке Claude (kb_release_v2_claude_layer_2026-05-17). Этот файл — основа для approval_queue v3.

---

## ТРЕБОВАНИЕ 7. Stage 6 fixtures и smoke-тест

У меня в исходной папке `kb_release_v2_claude_layer_2026-05-17/stage6_fixtures/` лежат 20 fixtures (10 FOTON + 10 UNPK) с конкретными `expected_in_draft` и `forbidden_in_draft`. Используй их как основу.

После повторной развёртки прогнать Stage 6 на тех же 20 диалогах. Ожидаемые метрики:
- Безопасностные (все 0): останутся 0
- `became_more_substantive`: должно вырасти с 7 до **15-18 из 20**. Это главная метрика готовности.
- `unsupported_numeric_promises`: 0 (если бот теперь знает цены, он не должен их выдумывать)
- `brand_separation_violation` (новая метрика): 0

Если `became_more_substantive` остался 7 — значит, факты по-прежнему не используются ботом, развёртка снова не доработана.

---

## ТРЕБОВАНИЕ 8. Не закрывать открытые вопросы самостоятельно

В `OPEN_QUESTIONS_FOR_TEAM.md` — 10 вопросов к РОПу/бухгалтерии/маркетингу/ИТ/лагерю.

Codex не должен:
- Делать догадки про значение этих фактов
- Помечать `status: verified` факты, привязанные к открытым вопросам
- Включать спорные факты в client_safe_text с `allowed_for_client_answer=true`

Все факты с привязкой к открытому вопросу должны иметь:
- `status: needs_owner_confirmation`
- `allowed_for_client_answer: false`
- `route_policy: manager_handoff_only` или `draft_for_manager`
- Поле `linked_open_question: "q14"` (или соответствующее)

---

## ТРЕБОВАНИЕ 9. Что НЕ менять

Эти куски v2 синхронизированы хорошо, не ломать:

`brand_rules.yaml`:
- `core_rule`
- `active_brand` (foton/unpk/unknown)
- `forbidden_client_mentions` (списки слов)
- `client_default_relationship_answer` (моя финальная формула)
- `pilot_mode.bot_answer_self_for_pilot`

`bot_policy.yaml`:
- `theme_routes` для matkap, tax_deduction, refund, complaint, intensive
- `payment_status.decision_matrix`
- `post_filter_draft_text`
- `unsupported_numeric_promise_detector`

Unit-тесты 60/60 в v2 прошли — их логику тоже не менять.

---

## КОНТРОЛЬНЫЙ СПИСОК ДЛЯ ВОЗВРАТА

Перед тем как отдать v3-handoff на ревью Claude, проверь:

- [ ] Все 16 пунктов из требования 5 отражены в facts_registry
- [ ] Цены 44600 / 74500 / 49000 / 82000 / 98000 / 75000 / 120000 / 89900 grep-находимы
- [ ] 0 записей с `client_safe_text` начинающимся «в УНПК / в Фотоне / у наших партнёров»
- [ ] 0 записей где `client_safe_text` содержит «Л035-» или другие номера лицензий
- [ ] 100% facts имеют `source_id`, который существует в source_registry
- [ ] approval_queue содержит 400+ пунктов, сгруппированных по 33 темам
- [ ] Stage 6 became_more_substantive ≥ 15 из 20
- [ ] Все 10 открытых вопросов — `status: needs_owner_confirmation`, `allowed_for_client_answer: false`
- [ ] brand_rules.yaml и bot_policy.yaml — без изменений по сравнению с v2

Если все ☑ — отдавай. Если что-то ☐ — итерация продолжается.
