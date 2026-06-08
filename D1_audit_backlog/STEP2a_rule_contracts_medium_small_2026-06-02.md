# Контракты правил (Шаг 2a): средние и мелкие семейства. 2026-06-02.

Автор: Клод 1. Источники прочитаны по сырью: `_produce_olympiad_online_template`, `_docs_safe_template`,
`_teacher_safe_template`, `_recordings_safe_template`, `_direct_process_safe_template`, `_matkap_safe_template`,
`_produce_tax_template` + их текст-константы (159-333). Сделаны подряд, без остановок (по просьбе Дмитрия).

Формат сжатый: эти семейства проще крупных и частично уже покрыты гейтом/A2.1. По эталону, но коротко.

---

## 1. olympiad — Олимпиадный онлайн

```yaml
rule_id: olympiad
title: Олимпиадная подготовка (Физтех онлайн)
intent: olympiad_inquiry
intent_subvariants: [price, grade_eligibility, vs_regular]
brand: unpk                              # Олимпиадная Физтех онлайн — УНПК
required_fact_keys:
  - olympiad.unpk.phystech.price         # UNPK_OLYMPIAD_PHYSTECH_PRICE_TEXT
  - olympiad.unpk.phystech.grades        # ТОЛЬКО 9 и 11 классы (память + CLAUDE.md)
data_rules:
  online_grades: [9, 11]                 # олимпиадная онлайн ТОЛЬКО 9 и 11 классы
  not_substitute_regular: true           # не подменять обычный курс олимпиадным и наоборот
  out_of_grade: manager_verify           # другой класс → менеджер проверит факт, не выдумывать группу
blocking_conditions:
  - grade_outside_9_11_promise           # подтверждение группы вне 9/11 без факта → handoff (UNPK_OLYMPIAD_PHYSTECH_HANDOFF_TEXT)
route_effect: bot_answer_self            # 9/11 + факт → ответ; иначе менеджер сверит
text_effect: composer_generates_from_data
preserve_exceptions:
  - не подтверждать класс/группу вне проверенного факта (checklist из кода)
  - не подменять обычный курс олимпиадным (scope: regular_online ≠ olympiad_online)
```
Негатив: 1) «олимпиадная для 7 класса?» → менеджер сверит, не выдумать группу. 2) обычный онлайн-курс не
выдавать как олимпиадный. ПОЗИТИВ: «олимпиадная Физтех 9 класс, цена?» → из факта.

---

## 2. docs — Договор / справка / лицензия / ПДн

```yaml
rule_id: docs
title: Документы (договор, справка для вычета, лицензия)
intent: documents_inquiry
intent_subvariants: [contract, certificate, license, legal_entity]
required_fact_keys:
  - docs.contract.timing                 # «в ближайшие дни»
  - docs.certificate.timing              # «до 10 дней, постараемся раньше» (CERTIFICATE_SAFE_TEXT)
  - docs.license.client_phrase           # «есть лицензия на образовательную деятельность» (без номеров)
data_rules:
  contract_timing: "в ближайшие дни"
  certificate_timing: "до 10 дней, постараемся раньше"
  certificate_type_not_specified: true   # тип справки НЕ уточнять без контекста (CLAUDE.md)
  license_no_numbers: true               # клиенту НЕ показывать номера лицензий/даты/юрлица
blocking_conditions:
  - pii_in_certificate_request           # ПДн + «справка/вычет» → PII_DOCUMENT_SAFE_TEXT (не эхо ПДн)
  - legal_entity_details                 # юрлицо/номер лицензии → CONTRACT_ENTITY_SAFE_TEXT (менеджер), не раскрывать
route_effect: bot_answer_self            # сроки договора/справки + «есть лицензия» — из факта
text_effect: composer_generates_from_data
preserve_exceptions:
  - НИКОГДА не показывать клиенту номер лицензии / дату / юрлицо (внутреннее) → «есть лицензия на образоват. деятельность»
  - ПДн в запросе справки → не повторять ПДн, безопасный ответ
  - тип справки без контекста не уточнять
```
Негатив: 1) «дайте номер лицензии» → «есть лицензия...», без номера. 2) клиент прислал ПДн → не эхо.
ПОЗИТИВ: «когда будет справка для вычета?» → «до 10 дней, постараемся раньше».

---

## 3. teacher — Преподаватели

```yaml
rule_id: teacher
title: Преподаватели (регалии без ФИО)
intent: teacher_inquiry
intent_subvariants: [general, specific_name, change_teacher, mendeleevo]
required_fact_keys:
  - teachers.regalia                     # «МФТИ, МГУ, ВШЭ, МИФИ, эксперты ЕГЭ, члены жюри» (TEACHERS_GENERAL)
data_rules:
  regalia_yes_names_no: true             # в ОБЩИХ вопросах давать регалии, НЕ ФИО (CLAUDE.md)
  name_depends_on_group: manager         # «как зовут / кто в Лобне» → имя зависит от группы → менеджер уточнит
blocking_conditions: []                  # не P0, просто дефер ФИО к менеджеру
route_effect: bot_answer_self            # регалии — сам; конкретное ФИО → менеджер уточнит по группе
text_effect: composer_generates_from_data
preserve_exceptions:
  - конкретное ФИО → «зависит от группы, менеджер уточнит» (TEACHERS_SPECIFIC), не выдумывать имя
  - «не понравился преподаватель» → TEACHERS_CHANGE (можно сменить, менеджер)
  - Менделеево/ЛВШ преподаватели → отдельный ответ (TEACHERS_MENDELEEVO)
```
Негатив: 1) «как зовут преподавателя физики в Лобне?» → не выдумать ФИО, менеджер уточнит. ПОЗИТИВ: «кто у
вас преподаёт?» → регалии (МФТИ/МГУ/ВШЭ...), без ФИО.

---

## 4. recordings — Записи занятий

```yaml
rule_id: recordings
title: Записи занятий (онлайн доступны, очные не ведутся)
intent: recording_inquiry
intent_subvariants: [online, offline]
required_fact_keys:
  - recordings.online.available          # «доступны записи / записи уроков доступны» (факт)
  - recordings.offline.not_recorded      # «запись очных занятий не ведётся» (факт)
data_rules:
  online_available: true                 # пропустил онлайн-урок → пересмотреть в записи
  offline_not_recorded: true             # очные занятия НЕ записываются
  format_decides: true                   # ответ зависит от формата (онлайн/очно) вопроса
brand_split: true                        # active_brand foton/unpk
route_effect: bot_answer_self
text_effect: composer_generates_from_data
preserve_exceptions:
  - очный формат → честно «запись очных не ведётся» (не обещать запись)
  - онлайн формат → «доступны, можно пересмотреть»
  - факт наличия записи ДОЛЖЕН быть в фактах (не выдумывать доступность)
```
Негатив: 1) «очные записываете?» → честно «нет» (не обещать). ПОЗИТИВ: «если пропущу онлайн-урок?» →
доступны в записи.

---

## 5. process — Процесс оформления / способ оплаты

```yaml
rule_id: enrollment_process
title: Как записаться / оформить / способ оплаты
intent: process_inquiry
intent_subvariants: [how_to_enroll, payment_method, fix_price_application]
required_fact_keys:
  - process.enrollment.steps             # как записаться/оформить
  - payment.methods                      # способы оплаты (счёт/банк/Долями) — пересечение с installment
data_rules:
  enrollment_via_manager: true           # оформление через менеджера/заявку
  payment_methods_link_to_installment: true  # «Долями/рассрочка» → правило installment (не дублировать)
blocking_conditions:
  - benign_hypothetical_refund           # гипотетический возврат внутри процесса → A2.1/refund-правило (успокоить остаток), не P0 авто
  - real_refund_claim                     # реальная претензия → P0 менеджер
route_effect: bot_answer_self
text_effect: composer_generates_from_data
preserve_exceptions:
  - «закрепить/зафиксировать цену по текущей заявке/оплате» → это price_fix-намерение, не общий процесс
  - способ оплаты «Долями/рассрочка» → отдать правилу installment (единый источник), не повторять
  - гипотетический возврат ≠ реальная претензия (refund-правило различает)
```
Негатив: 1) «хочу вернуть деньги, недоволен» → P0 менеджер (не процессный ответ). ПОЗИТИВ: «как записаться?»
→ шаги из факта.

---

## 6. matkap — Материнский капитал

```yaml
rule_id: matkap
title: Материнский капитал (федеральный)
intent: matkap_inquiry
intent_subvariants: [federal, regional, sfr_approval]
required_fact_keys:
  - matkap.federal.accepted              # федеральный принимаем (MATKAP_FEDERAL_TIMING)
  - matkap.regional.not_accepted         # региональный НЕ принимаем (MATKAP_REGIONAL)
data_rules:
  federal_yes: true                      # работаем с федеральным маткапиталом
  regional_no: true                      # региональный не принимаем → менеджер подскажет порядок
  sfr_no_guarantee: true                 # одобрение — СФР, не обещать одобрение (MATKAP_SFR_REVIEW)
  brand_aware: true                      # можно и для Фотона, и для УНПК, с учётом active_brand
blocking_conditions:
  - sfr_approval_promise                 # обещание одобрения СФР → block (рассмотрение проводит СФР)
route_effect: bot_answer_self            # справочно из факта; одобрение/региональный → менеджер
text_effect: composer_generates_from_data
preserve_exceptions:
  - НЕ обещать одобрение СФР (checklist кода)
  - региональный маткапитал НЕ принимать (только федеральный)
  - active_brand: маткапитал валиден для обоих брендов, но в рамках текущего
```
Негатив: 1) «точно одобрят маткапитал?» → не обещать (решает СФР). 2) «региональный примете?» → нет, только
федеральный. ПОЗИТИВ: «можно оплатить федеральным маткапиталом?» → да, справочно из факта.

---

## 7. tax — Налоговый вычет

```yaml
rule_id: tax
title: Налоговый вычет (КНД 1151158)
intent: tax_inquiry
intent_subvariants: [amount, how_to_form, fns_decision, license]
required_fact_keys:
  - tax.amount                           # сумма/13%/лимит (TAX_AMOUNT)
  - tax.how_to_form                      # как оформить/подать (TAX_ONLINE_FORM)
  - tax.license                          # «есть лицензия» без номеров (TAX_LICENSE)
data_rules:
  fns_decides: true                      # ФНС рассматривает и решает — не гарантировать возврат (TAX_FNS_REVIEW)
  license_no_numbers: true               # лицензия без номеров (как docs)
  certificate_helps: true                # справка подтверждает обучение
  brand_aware: true                      # вычет валиден для Фотона и УНПК, с учётом active_brand
blocking_conditions:
  - fns_return_guarantee                 # гарантия возврата от ФНС → block (TAX_FNS_REVIEW вместо обещания)
route_effect: bot_answer_self            # сумма/как оформить/лицензия — справочно; гарантия ФНС → дефер
text_effect: composer_generates_from_data
preserve_exceptions:
  - НЕ гарантировать возврат от ФНС (решает ФНС)
  - лицензия без номеров (как docs.license)
  - сумма вычета: справочно (13%, лимит), без обещания одобрения
```
Негатив: 1) «точно вернут 13%?» → ФНС решает, не гарантировать. 2) «номер лицензии для вычета?» → «есть
лицензия», без номера. ПОЗИТИВ: «как оформить вычет?» → шаги из факта + справка помогает.

---

## ИТОГ Шага 2a (все семейства разобраны)

**Крупные (8 семей, отдельные файлы):** installment(эталон), discount, price, camp_lvsh, schedule, trial,
terminal→platform_access(+безопасность в гейт), format_choice.
**Средние/мелкие (7 семей, этот файл):** olympiad, docs, teacher, recordings, enrollment_process, matkap, tax.

**Итого 15 семейств.** Все доменные функции монолита разложены в декларативные правила одного формата:
`intent(+subvariants→planner)` + `required_fact_keys` + `data_rules` + `blocking_conditions(→гейт)` +
`route_effect` + `text_effect=composer_generates_from_data` + `preserve_exceptions` + негатив-тесты.

**Доказанные модели переноса (весь спектр):**
1. данные-в-коде + функция (installment, discount) — значения константны;
2. правила + KB-факты (price, recordings, olympiad) — значения динамичны из базы;
3. + live-данные → менеджер (camp seats) — значения НЕТ, динамика CRM;
4. + Tallanto-дериватив + скрытое исключение-как-правило (schedule, контакт-часы≠дни);
5. безопасность, замаскированная под домен (terminal) → в гейт, не в правила;
6. справочные с внешним решателем (matkap→СФР, tax→ФНС) — отвечаем справку, решение не обещаем.

**Главное открытие Шага 2a:** карта (Шаг 0) местами неточна (terminal помечен domain, а он 80% безопасность).
Поэтому миграция Кодекса = читать каждую функцию + негатив-тест, метке не доверять.

**Следующий шаг:** свести 15 контрактов в ЕДИНЫЙ машинный реестр правил (`rules_registry.yaml` —
deliverable Шага 2a для Кодекса), затем — регрейд Прогона 1 (гейт, обе половины M1+Кодекс) и Шаг 2b (миграция).
```
