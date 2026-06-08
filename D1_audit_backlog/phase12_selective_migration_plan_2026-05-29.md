# Phase 12 — селективный план миграции safety-шаблонов из legacy в v2

Автор: второй Claude. Дата: 2026-05-29. Read-only. База: `subscription_llm_deep_audit_2026-05-29.md`. Цель Phase 12: hard_gate=0 как стандарт пилота. НЕ пишу код, НЕ переписываю v2-архитектуру.

## Round-trip памяти между ходами — ответ (read-only)

Что подтверждено по коду:
- В пилотном пути `telegram_pilot_context_builder.build_telegram_pilot_context` (`:102`) контекст для `classify_answer_safety`/intent/драфтера строится из **`memory.to_prompt_view()`** (`:136/156/160/221/261`) — то есть **УРЕЗАННОГО** представления, не `to_json_dict`.
- `to_prompt_view` (`dialogue_memory.py:199`) СОХРАНЯЕТ: полный `p0_latch` (`:215`), `client_confirmed_slots`/`crm_known_slots`/`bot_inferred_slots`/`do_not_reask_slots` (`:221-224`), `held_state.to_prompt_view` (`:225`), `recent_turns[-20:]`, сводки. ТЕРЯЕТ vs `to_json_dict`: у `known_slots` остаются только ЗНАЧЕНИЯ без `source`/`confidence` (`:206`), и rolling-списки урезаны `[-5:]/[-8:]` (vs `[-12:]`/полные в `to_json_dict`).
- Контекст-билдер наружу отдаёт `dialogue_memory_view = to_prompt_view` (`:261`), полный `to_json_dict` он НЕ эмитит.

**Вывод:** эффективное межходовое состояние, которое видят классификатор/драфтер, — **урезанный `to_prompt_view`**, НЕ полный `to_json_dict`. Латч и confirmed/crm-слоты переживают ход (хорошо); `known_slots` теряют source/confidence; старые элементы списков выпадают.

**Остаток (честно):** долговременный СТОР между ходами (что именно рантайм кладёт в `previous_memory` на следующий ход — полный объект или этот же урезанный view) задаётся вызывающим рантаймом ВНЕ `channels/` (бот-хендлер/хранилище); read-only я его не локализовал (`build_telegram_pilot_context` вызывается из слоя выше `channels/`). Для Phase 7/12 нужно, чтобы стор сохранял `to_json_dict` (с source/confidence латча и слотов). **ОТКРЫТО для Кодекса:** подтвердить формат durable-стора. Практический ориентир сейчас: считать состояние урезанным.

---

## §1. Полный список шаблонов (каскад `apply_subscription_policy_guards`, ~2120-2750)

Все вызываются ТОЛЬКО в legacy; в v2-цепочке (`:1002-1029`) НЕ вызываются → мертвы в пилоте. Порядок = порядок применения (первый сработавший ставит `metadata[...]=True`, остальные себя пропускают).

| # | template (флаг) | def / строка применения | SAFE_TEXT | Что защищает |
|---|---|---|---|---|
| 1 | unpk_zvsh_waitlist | `:2265` | `UNPK_ZVSH_WAITLIST_SAFE_TEXT:187` | ЗВШ Менделеево — нет дат, лист ожидания (closed-world дат) |
| 2 | cross_brand | `_cross_brand_safe_template:4810` / `:2273` | `CROSS_BRAND_GENERIC/LICENSE/PLATFORM:335-337` | **Бренд-разделение, лицензия, платформа** |
| 3 | direct_process | `_direct_process_safe_template:4466` / `:2301` | `PRICE_FIX_PROCESS:338` и др. | Процесс фикс-цены/записи |
| 4 | payment_method | `_payment_method_safe_template:4510` / `:2305` | `PAYMENT_BANK_TRANSFER:153` | Способ оплаты (перевод/счёт) |
| 5 | terminal | `_terminal_safe_template:4663` / `:2315` | ADDRESS_*/CONTACT_*/`IDENTITY_*:348-356`/OFF_TOPIC_*/`SOFT_NEGATIVE:93` | **Природа бота (identity)** + адрес/контакты/офф-топик/мягкий отказ |
| 6 | matkap | `_matkap_safe_template:4446` / `:2370` | `MATKAP_REGIONAL/SFR/FEDERAL:192-194` | Не обещать одобрение СФР, не брать региональный |
| 7 | tax | `_tax_safe_template:5421` / `:2378` | `TAX_*:199-211` | Не гарантировать возврат ФНС; лицензия |
| 8 | camp | `_camp_safe_template:5447` / `:2386` | `*_LVSH_*`, `*_CAMP_*:212-327` | ЛВШ/лагерь: цены/места/даты/проживание |
| 9 | format_choice | `_format_choice_safe_template:5577` / `:2401` | (онлайн/очно) | Выбор формата |
| 10 | olympiad_online | (def ~5xx) / `:2416` | (олимпиадный онлайн) | **Не подтверждать класс, которого нет в факте** (E7/C2) |
| 11 | price_installment_multitopic | `_price_installment_multitopic_safe_template:4525` / `:2437` | — | Цена+рассрочка в одном вопросе |
| 12 | installment | `_installment_safe_template:4117` / `:2453` | `FOTON_INSTALLMENT*:126-153` | Рассрочка/Долями: числа/условия |
| 13 | pricing | `:2483` | `*_PRICE_SAFE_TEXT:166-176` | Ценовые числа без факта |
| 14 | schedule_frequency | `_schedule_frequency_safe_template:4373` / `:2499` | — | Частота/расписание (vs часы связи, E3) |
| 15 | schedule_confirmation | `_schedule_confirmation_safe_template:4403` / `:2515` | `UNPK_FORMAT_OR_DAYS:321` | Подтверждение расписания |
| 16 | discount | `_discount_safe_template:4025` / `:2533` | `DISCOUNT_STACKING:125` | Скидки: % и несуммирование |
| 17 | trial | `_trial_safe_template:5720` / `:2564` | `*_TRIAL_*:227-269` | Пробное/фрагмент занятия |
| 18 | recordings | `_recordings_safe_template:4303` / `:2590` | — | Записи занятий |
| 19 | admission_guarantee | `:2597` | `ADMISSION_GUARANTEE:101` | **Не гарантировать поступление** |
| 20 | result_guarantee | `:2603` | `RESULT_GUARANTEE:97` | **Не гарантировать балл/результат** (≈ «N баллов») |
| 21 | docs | `_docs_safe_template:4840` / `:2620` | `CERTIFICATE/PII_DOCUMENT:329-330` | Справки/PII-документы |
| 22 | teacher | `_teacher_safe_template:4878` / `:2640` | `TEACHERS_*:331-334` | Преподаватели: регалии без ФИО |

«~30» = 22 селектора + terminal раскрывается в ~12 терминальных текстов (адрес/контакт/identity/off-topic ×2 бренда). Помимо них в v2 УЖЕ есть (мигрировать НЕ надо): `unstated_subject`, `unsupported_promise`, `brand_separation`, `payment_confirmation`, `unconfirmed_operational` — они в v2-цепочке.

---

## §2. Категории критичности для пилота

**P0 — блокер пилота (без них возможны hard_gate / brand-leak / юр.риск):**
- **cross_brand** (#2) — брэнд-leak. Хотя в v2 есть `brand_separation` guard, шаблон даёт безопасный текст-ответ; проверить, покрывает ли v2-guard полноту (ОТКРЫТО). Считаю P0.
- **terminal→identity** (#5, часть IDENTITY_*) — раскрытие природы бота (мета-leak). P0.
- **olympiad_online** (#10) — класс-scope; прямой механизм против E7/C2 hard_gate. P0.
- **result_guarantee** (#20) и **admission_guarantee** (#19) — обещание балла/поступления = юр.риск + ровно класс «N баллов». P0.
- **matkap/tax — часть «не гарантировать СФР/ФНС»** (#6/#7) — финансово-юр. обещание. P0 (именно guarantee-блок; справочная часть — P1).

**P1 — важно, не блокер:**
- camp (#8), format_choice (#9), pricing (#13), installment (#12), price_installment_multitopic (#11), discount (#16) — качество/числа; частично прикрыты v2-guard `unsupported_promise`.
- schedule_frequency/confirmation (#14/#15) — расписание vs часы связи (E3); частично закрыто фиксом `_mentions_contact_hours` (Волна 0).
- trial (#17), recordings (#18), docs (#21), teacher (#22), payment_method (#4), direct_process (#3) — справочные/процедурные.
- terminal→address/contact/off-topic (#5 инфо-часть) — удобство, не безопасность.

**P2 — можно отложить / не мигрировать:**
- unpk_zvsh_waitlist (#1) — редкий трафик, лист ожидания.
- Любые шаблоны по модульным М9/М11 (discontinued по CLAUDE.md), промокодам (убраны из клиентского слоя), `OLD_TERM`/`FORCED_DISCOUNT` — см. §7.

---

## §3. Связь с конкретными провалами

- **E7_olymp_04 (Физтех, 6 класс, ОЧНО).** Ближайший шаблон — **olympiad_online** (#10), его смысл «не подтверждать класс, которого нет в проверенном факте». НО он ONLINE-специфичен; E7 — очное → **текущий шаблон НЕ покрывает очную олимпиаду.** Нужен либо расширенный olympiad-scope (онлайн+очно), либо **новый** grade-scope guard. + зависит от KB-факта о классах олимпиады (Phase 2, `applies_to.grades`). Вердикт: **P0, частично существующий (online) + завести offline/grade-scope.**
- **C2_multi_subj_03 (подмена базовой цены онлайн олимпиадной Физтех).** Прямо закрывается **olympiad_online** (#10) — «не подтверждать олимпиадный продукт как ответ на обычную цену». Миграция #10 в v2 помогает напрямую. Остаток — KB-пробел (онлайн-цены УНПК нет в client-safe, см. мой kb-анализ) — это Phase 2, не шаблон. Вердикт: **P0, шаблон есть, мигрировать.**
- **S3_vyezdnaya_04 (closed-world «других выездных нет»).** **camp** (#8) частично (не путать ЛВШ с городским лагерем), но closed-world-утверждение «нет других» НИ ОДИН шаблон не закрывает. Нужен **новый** closed-world-enforcement (требует отрицательного факта в KB — Phase 2.3/Phase 7.3). Вердикт: **P0, нужно завести новый** (camp #8 — вспомогательно, P1).

Итог §3: из 3 hard_gate один (C2) закрывается миграцией существующего шаблона; E7 — частично (нужно offline-расширение + KB); S3 — нужен новый guard + KB. То есть hard_gate=0 ≠ только миграция: 2 из 3 требуют ещё KB-полей (Phase 2). Это важно для срока.

---

## §4. Сложность миграции

Все шаблоны — чистые функции `_X_safe_template(result, client_message, context) -> str|""`. Базовая миграция = вызвать в v2-цепочке (после контентных guard'ов, перед sanitize) + сохранить precedence-флаги.

- **Trivial (1-2 ч):** result_guarantee (#20), admission_guarantee (#19) — флаг-основанные, без сложной логики; matkap/tax guarantee-блок (#6/#7); cross_brand (#2) — самостоятельный.
- **Medium (полдня-день):** terminal (#5) — раскрывается в подвыбор (identity/address/contact/off-topic), мигрировать как один диспетчер; olympiad_online (#10) — online-логика + интерфейс к scope; pricing/installment/discount-кластер (#11-13,16) — нужна развязка с v2-guard `unsupported_promise` (чтобы не дублировать/конфликтовать); schedule_* (#14/15); camp (#8); trial (#17); docs/teacher (#21/22).
- **Complex (несколько дней):** offline-олимпиада grade-scope (новый, E7) и closed-world enforcement (новый, S3) — требуют KB-полей (Phase 2: `applies_to.grades`, отрицательные факты) и новой логики, не просто копирование. + общий **диспетчер precedence** (заменить ручные `or metadata.get(...)` на один «выбери максимум один шаблон по приоритету») — Complex, но окупается.

---

## §5. Зависимости и конфликты

**Зависимости (мигрировать вместе):**
- Ценовой кластер: pricing (#13) + installment (#12) + price_installment_multitopic (#11) + discount (#16) — общая тема цены/чисел.
- terminal (#5) — address/contact/identity/off-topic как один блок.
- camp (#8) ↔ format_choice (#9) ↔ olympiad_online (#10) — общая product-scope логика; olympiad-scope опирается на scope из `fact_scope_spec`.
- Все шаблоны гейтятся `cross_brand_guarded()` — нужен этот helper в v2 (в v2 есть `brand_separation` guard; убедиться, что `cross_brand_guarded()` доступен/эквивалентен).

**Конфликты (нельзя мигрировать «в лоб» оба без развязки):**
- pricing/discount/result_guarantee (числа) ↔ v2-guard `unsupported_promise` (`:1019`) — оба трогают числа/обещания; без приоритета получим двойную перезапись. Решить: guard проверяет, template переписывает — порядок «template до unsupported_promise» или взаимное исключение по флагу.
- terminal→identity (#5) ↔ существующий `guard_identity_disclosure` (в v2 — ОТКРЫТО, см. аудит) — если identity-guard уже в v2, шаблон identity дублирует; мигрировать только недостающую часть.
- schedule_* (#14/15) ↔ фикс `_mentions_contact_hours` (Волна 0, `fact_claim_audit`) — частичное пересечение; согласовать, чтобы не было двойного wrong_scope.

---

## §5.5. Сводная матрица миграции (решение по каждому шаблону)

| template | приоритет | сложность | зависит от | конфликтует с | закрывает провал |
|---|---|---|---|---|---|
| cross_brand (#2) | **P0** | Trivial | brand_separation guard / `cross_brand_guarded()` | v2 brand_separation (проверить полноту) | brand-leak |
| terminal→identity (#5) | **P0** | Medium | terminal-диспетчер | v2 guard_identity_disclosure (ОТКРЫТО) | мета-leak природы бота |
| olympiad_online (#10) | **P0** | Medium | fact_scope (`regular_online`↔`olympiad_online`) | unsupported_promise | **C2** (online), частично E7 |
| offline/grade-scope олимпиады | **P0 (новый)** | Complex | Phase 2 `applies_to.grades` | olympiad_online | **E7** (очное) |
| closed-world enforcement | **P0 (новый)** | Complex | Phase 2 отрицательные факты | camp | **S3** |
| result_guarantee (#20) | **P0** | Trivial | — | unsupported_promise («N баллов») | обещание балла, hard_gate-класс |
| admission_guarantee (#19) | **P0** | Trivial | — | — | обещание поступления |
| matkap-guarantee (#6) | **P0** | Trivial | — | — | обещание СФР |
| tax-guarantee (#7) | **P0** | Trivial | — | — | обещание ФНС |
| camp (#8) | P1 | Medium | format_choice | closed-world (новый) | ЛВШ/лагерь подмена (S3 вспом.) |
| schedule_frequency (#14) | P1 | Medium | — | `_mentions_contact_hours` (Волна 0) | E3 (часы связи как расписание) |
| schedule_confirmation (#15) | P1 | Medium | schedule_frequency | — | расписание |
| pricing (#13) | P1 | Medium | ценовой кластер | unsupported_promise | цена без факта |
| installment (#12) | P1 | Medium | ценовой кластер | unsupported_promise | рассрочка-числа |
| price_installment_multitopic (#11) | P1 | Medium | ценовой кластер | pricing/installment | цена+рассрочка |
| discount (#16) | P1 | Medium | ценовой кластер | unsupported_promise | скидки-числа |
| format_choice (#9) | P1 | Medium | camp/olympiad | — | выбор формата |
| trial (#17) | P1 | Medium | — | — | пробное |
| recordings (#18) | P1 | Trivial | — | — | записи |
| docs (#21) | P1 | Trivial | — | — | справки/PII |
| teacher (#22) | P1 | Trivial | — | — | ФИО преподавателя |
| matkap/tax справочная (#6/#7) | P1 | Trivial | guarantee-часть | — | инфо вычет/маткап |
| payment_method (#4) | P1 | Trivial | — | — | способ оплаты |
| direct_process (#3) | P1 | Trivial | — | — | процесс фикс-цены |
| terminal→address/contact/off-topic (#5) | P1 | (в составе terminal) | terminal | — | справка адрес/контакт |
| unpk_zvsh_waitlist (#1) | P2 | Trivial | — | — | ЗВШ лист ожидания |
| promocode/forced_discount/M9-M11/OLD_TERM | НЕ мигрировать | — | — | — | устарело (см. §7) |

Чтение матрицы: P0-строк — **9** (включая 2 новых, требующих KB Phase 2). Именно они = «ready for pilot». P1 — ~14. P2/не-мигрировать — остальное.

## §6. Roadmap миграции

**Критерий «Phase 12 ready for pilot» (hard_gate=0 минимум):**
Закрыты все классы, дающие hard_gate/brand-leak/юр.риск в v2:
1. cross_brand (#2) — нет брэнд-leak.
2. terminal→identity (#5) — нет раскрытия природы бота.
3. result_guarantee (#20) + admission_guarantee (#20/19) — нет обещаний балла/поступления.
4. olympiad_online (#10) + **новый offline/grade-scope** — закрывает E7/C2 (с KB `applies_to.grades`).
5. **новый closed-world enforcement** — закрывает S3 (с отрицательным фактом KB).
6. matkap/tax guarantee-блок (#6/#7).
+ диспетчер precedence (чтобы мигрированные шаблоны не конфликтовали).

**Порядок и оценки (с приоритизацией — путь к 2-3 неделям):**
- **Неделя 1 (P0, миграция существующих):** cross_brand (Trivial), result/admission_guarantee (Trivial), matkap/tax-guarantee (Trivial), terminal-identity (Medium), olympiad_online online (Medium) + каркас диспетчера precedence (Medium). → закрывает brand-leak, identity, обещания, C2.
- **Неделя 2 (P0, новое + KB-зависимое):** offline/grade-scope олимпиады (Complex, нужен Phase 2 `applies_to.grades`) + closed-world enforcement (Complex, нужен отрицательный факт KB) → закрывает E7, S3. ⚠️ эти двое зависят от Phase 2 KB-полей — если KB v6.4 не готов, hard_gate=0 не достигается только шаблонами.
- **Неделя 2-3 (критичный P1):** schedule_* (E3-смежное), camp, ценовой кластер (с развязкой от `unsupported_promise`), trial/docs/teacher. → качество, не блокер.

**Без приоритизации (≈2 месяца):** миграция всех ~22+12 шаблонов + полный диспетчер + тесты идемпотентности + регресс-suite на стек (см. аудит). С приоритизацией P0-набор + критичный P1 = **2-3 недели** (совпадает с оценкой Дмитрия), при условии что Phase 2 KB-поля (grade-scope, отрицательные факты) готовы параллельно.

---

## §7. Что НЕ мигрировать

- **Дубликаты v2-guard'ов:** `unstated_subject`, `unsupported_promise`, `brand_separation`, `payment_confirmation`, `unconfirmed_operational` — уже в v2-цепочке; не дублировать.
- **Устаревшее по CLAUDE.md:** любые шаблоны про модульные М9/М11 (discontinued); промокоды (`PROMOCODE_SAFE_TEXT:180`, `FORCED_DISCOUNT_SAFE_TEXT:105`) — промокоды убраны из клиентского слоя → НЕ мигрировать в клиентский путь (оставить как manager-only при необходимости).
- **`OLD_TERM_SAFE_TEXT:364`** — переходный шаблон для старых форматов; проверить актуальность, вероятно не нужен.
- **Мёртвые/недостижимые ветки** terminal (off-topic generic, если в v2 off-topic решается иначе) — мигрировать только реально достижимые.
- unpk_zvsh_waitlist (#1) — P2, низкий трафик, отложить.

> Перед удалением «устаревшего» — подтвердить у Дмитрия/Кодекса, что v2-логика реально заменяет (а не что шаблон просто не вызывался). Не выдумывать deprecated-статус.

## Открытые вопросы

1. Durable-стор памяти между ходами (полный `to_json_dict` или урезанный `to_prompt_view`) — вне `channels/`, подтвердить Кодексу (критично для Phase 7 и для переноса латч-состояния).
2. Входит ли `guard_identity_disclosure` уже в v2-цепочку (через `apply_input_policy_guards`)? Если да — terminal-identity мигрировать частично. (Аудит, ОТКРЫТО.)
3. Точная def-строка `_olympiad_online_safe_template` и охватывает ли он только online (по коду — да; подтвердить для оценки offline-расширения).
4. Полнота v2-guard `brand_separation` vs шаблон cross_brand — нужен ли шаблон, если guard уже даёт безопасный ответ.
5. Готовность Phase 2 KB-полей (`applies_to.grades`, отрицательные факты) — от неё зависит, достижим ли hard_gate=0 (E7/S3) в срок 2-3 недели только шаблонами.
6. Идемпотентность мигрированных шаблонов в v2 (v2 делает re-verify после каждого guard'а) — проверить, что повторный прогон не осциллирует.
