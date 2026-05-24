# ТЗ: Answer Quality Rewriter для Telegram-ботов Фотона и УНПК МФТИ

Дата: 2026-05-23.

Статус: ТЗ на реализацию, реализация отдельным следующим шагом.

## 1. Контекст

После внедрения базы знаний, brand isolation, policy C, gold/few-shot и P0-защит прогнаны:

- `v8_targeted16_2026-05-22.jsonl`;
- `MEGA_autonomy_tests_v6_2026-05-22.jsonl`;
- `MEGA_multitopic_batch_v5_2026-05-22.jsonl`.

Итоги последнего среза:

- `v8_targeted16`: 16/16 без hard FAIL, но все 16 `PASS_WITH_NOTES`.
- Hard-gates: 0 нарушений.
- Средний тон: 48.1/100.
- Soft flags:
  - `ignored_question`: 13/16;
  - `templated_opening`: 13/16;
  - `over_handoff`: 12/16;
  - `invited_to_visit`: 2/16.
- Статичный v6/v5: 46 PASS / 16 FAIL; forbidden-утечек в FAIL нет.

Официальный baseline для этого блока:

- `audits/_inbox/telegram_dynamic_client_sim_v8_targeted16_20260523_145732/` — актуальный targeted16 baseline: `0 FAIL / 16 PASS_WITH_NOTES`;
- `audits/_inbox/telegram_static_v6_v5_quick_20260523_152353/` — актуальный static baseline: `46 PASS / 16 FAIL`;
- старые отчёты с `1 FAIL / brand_leak` по `МТС Линк` считаются устаревшими для этого блока: `МТС Линк / Webinar` — общая платформа, не брендовая утечка.

Главный вывод: проблема уже не в количестве фактов и не в P0-безопасности. Факты часто есть, но бот не умеет стабильно использовать их во втором и последующих ходах диалога.

Важное уточнение после ревью Claude: последний прогон — это baseline до полноценного слоя качества ответа. В проекте уже синхронизированы policy C, gold/few-shot корпус и часть памяти диалога, но это ещё не полноценный контроллер поведения:

- few-shot уже попадает в prompt как мягкая подсказка;
- `known_dialog_fields` и `recent_messages` уже частично попадают в dynamic sim;
- live Telegram runtime уже имеет `funnel_state` и `known_slots`;
- но dynamic sim и live runtime ещё нужно явно синхронизировать;
- rewriter ещё не реализован;
- v6/v5 частично содержат устаревшие ожидания маршрута после ослабления автономности.

Поэтому первый шаг реализации — не “сразу переписывать всё вторым LLM-вызовом”, а сначала добить первый проход: память диалога, answer-first, few-shot и одинаковый контекст между live и тестовым раннером.

## 2. Цель

Сделать отдельный слой качества клиентского ответа:

`answer_quality_rewriter`

Он должен после первичной генерации проверить, отвечает ли бот на последний прямой вопрос клиента, учитывает ли уже известные данные, не уходит ли без нужды к менеджеру и не звучит ли как шаблон. Если ответ слабый, слой должен переписать его в более полезный и живой вариант, сохранив P0/brand/fact safety.

Целевая формула ответа:

1. Прямой ответ на последний вопрос клиента.
2. Учёт уже известных данных: класс, предмет, формат, бренд, контекст диалога.
3. Проверенный факт, если он есть.
4. Если факт частичный или отсутствует — честный полезный ориентир без выдумки.
5. Один следующий шаг.
6. Тёплый, человеческий тон без канцелярита.

## 3. Не цели

В этом блоке не делаем:

- новые факты базы знаний;
- новые коммерческие решения РОПа;
- полную замену prompt builder;
- live-write в AMO/Tallanto;
- отправку сообщений клиентам;
- полный v8;
- большой рефакторинг Telegram-бота;
- попытку “обучить модель” на уровне fine-tuning.

## 4. Ключевой принцип безопасности

`answer_quality_rewriter` не имеет права ослаблять P0-защиту.

Приоритеты:

1. P0/brand/fact safety.
2. Не выдумывать.
3. Ответить прямо.
4. Продвинуть к следующему шагу.
5. Сделать ответ живым и тёплым.

Если между “ответить прямо” и “не выдумывать” конфликт, побеждает “не выдумывать”.

Rewrite запрещён полностью для:

- любого `route=manager_only`;
- любого `message_type=manager_only`;
- любого результата с флагами `high_risk_*`, `combined_high_risk_manager_only`, `refund_zero_collect`, `legal_threat`, `payment_confirmation_blocked`, `brand_separation_blocked`;
- жалоб, возвратов, юридических угроз, спорной оплаты, договорных претензий, документов, СФР/ФНС-статусов и любых тем, где клиент ожидает управленческого действия;
- многотемного сообщения, где хотя бы одна часть P0/high-risk.

В этих случаях слой качества может только добавить metadata/finding о качестве, но не переписывать клиентский текст и не повышать маршрут.

## 5. Где встраивать в текущую архитектуру

Перед встраиванием rewriter обязательно синхронизировать контекст live Telegram и dynamic sim. Иначе мы будем тестировать один контур, а сотрудники — другой.

Обязательный контракт контекста:

- `recent_messages`;
- `known_dialog_fields`;
- `known_client_fields`;
- `known_slots`;
- `funnel_state`;
- `missing_slots`;
- `next_best_question`;
- `next_step_type`;
- `answer_quality_reference`;
- `few_shot_style_examples`;
- `few_shot_correction_examples`;
- `confirmed_facts`;
- `facts_context`;
- `knowledge_snippets`;
- `active_brand`.

`active_brand` неизменяем в рамках одного ответа. Rewriter получает только факты активного бренда и не имеет права сам добавлять или выводить факты другого бренда. Few-shot и gold-примеры бренд-изолированы и не считаются источником фактов: любую цену, дату, скидку, место, наличие мест, фиксацию цены, документ или действие можно писать только если это выводится из `confirmed_facts`, `facts_context`, safe template или явно проверенного контекста активного бренда.

Если какое-то поле есть в live Telegram runtime, но отсутствует в dynamic sim, targeted16 не считается честной проверкой live-бота.

Текущий путь:

`SubscriptionLlmDraftProvider.build_draft()`

Сейчас примерно:

1. `build_draft_prompt()`;
2. LLM draft;
3. `apply_payment_confirmation_guard`;
4. `apply_brand_separation_guard`;
5. `apply_input_policy_guards`;
6. `apply_high_risk_content_guards`;
7. `apply_unsupported_promise_guard`;
8. `apply_unconfirmed_operational_specificity_guard`;
9. `apply_known_context_redundant_question_guard`;
10. `apply_funnel_policy_guard`;
11. `apply_autonomy_matrix_guard`.

Новый путь:

1. `build_draft_prompt()`;
2. LLM draft с усиленным первым проходом:
   - context parity между live и dynamic;
   - answer-first инструкции;
   - few-shot/gold как эталон формы;
   - `known_slots`/`funnel_state` как источник памяти;
   - запрет повторного вопроса об уже известных данных.
3. deterministic answer-quality assessment:
   - только флаги и metadata;
   - без изменения текста;
   - нужен, чтобы понять, требуется ли rewrite.
4. hard guards первой линии:
   - payment confirmation;
   - brand separation;
   - input policy;
   - high risk;
   - unsupported promise;
   - unconfirmed operational specificity;
   - known context redundant question;
   - funnel policy.
5. `apply_answer_quality_rewriter` только если assessment нашёл rewrite-флаг.
6. hard guards второй линии:
   - brand separation;
   - high risk;
   - unsupported promise;
   - unconfirmed operational specificity;
   - known context redundant question;
   - funnel policy.
7. `apply_autonomy_matrix_guard`.

Причина второй линии guards: rewriter может улучшить текст, но после него всё равно нужно снова проверить P0, бренд, выдумки, CRM/сроки и повтор известных данных.

Финальные guards должны проверять не только `draft_text`, но и metadata: route, safety_flags, used facts, context warnings и флаги `answer_quality_*`.

## 6. Новый модуль

Создать:

`src/mango_mvp/channels/answer_quality_rewriter.py`

Основные сущности:

```python
@dataclass(frozen=True)
class AnswerQualityFinding:
    code: str
    severity: str  # blocker | rewrite | note
    reason: str
    evidence: str = ""

@dataclass(frozen=True)
class AnswerQualityAssessment:
    passed: bool
    needs_rewrite: bool
    direct_question: str
    known_slots: Mapping[str, str]
    answerable_parts: tuple[str, ...]
    findings: tuple[AnswerQualityFinding, ...]
    rewrite_instruction: str

@dataclass(frozen=True)
class AnswerQualityRewriteResult:
    result: SubscriptionDraftResult
    assessment: AnswerQualityAssessment
    rewritten: bool
    rewrite_provider: str
```

Публичные функции:

```python
def assess_answer_quality(
    result: SubscriptionDraftResult,
    *,
    client_message: str,
    context: Mapping[str, Any] | None,
) -> AnswerQualityAssessment:
    ...

def apply_answer_quality_rewriter(
    result: SubscriptionDraftResult,
    *,
    client_message: str,
    context: Mapping[str, Any] | None,
    rewrite_runner: AnswerRewriteRunner | None = None,
) -> SubscriptionDraftResult:
    ...
```

## 7. Что проверяет `assess_answer_quality`

### 7.1. Ответил ли бот на последний прямой вопрос

Нужно определить последний прямой вопрос клиента по текущему сообщению и recent messages.

Примеры:

- “это цена прямо на сейчас?”
- “можно зафиксировать?”
- “там проценты есть?”
- “это через банк?”
- “Долями на 4 платежа?”
- “приезжать никуда не надо?”
- “что входит в лагерь?”
- “проживание и питание отдельно?”
- “места есть?”
- “фрагмент бесплатный?”

Коды finding:

- `ignored_direct_question`
- `answered_nearby_topic`
- `lost_followup_context`

### 7.2. Не спросил ли уже известное

Источники известных данных:

- `known_dialog_fields`;
- `known_client_fields`;
- `known_slots`;
- `funnel_state`;
- recent messages;
- debug-client context.

Поля:

- класс;
- предмет;
- формат;
- бренд;
- цель;
- телефон;
- имя, если клиент сам написал его в текущем диалоге.

Коды finding:

- `reasked_known_grade`
- `reasked_known_subject`
- `reasked_known_format`
- `reasked_known_identity`

### 7.3. Не ушёл ли к менеджеру при наличии факта

Если:

- тема зелёная;
- нет P0;
- active brand известен;
- факт есть в `confirmed_facts`, `facts_context`, `knowledge_snippets` или safe template;
- текущий ответ содержит только “менеджер проверит / уточнит / свяжется”,

то это finding:

- `over_handoff_with_verified_fact`

Важно: не каждое упоминание менеджера плохо. Плохо, если менеджер заменяет полезный ответ.

Допустимо:

> ЛВШ стоит 93 100 ₽, полная стоимость 98 000 ₽. Наличие мест по вашей смене проверит менеджер.

Плохо:

> Менеджер проверит стоимость и свяжется.

### 7.4. Есть ли один следующий шаг

Ответ должен завершаться одним понятным шагом:

- “Напишите класс — подберу подходящий формат.”
- “Если онлайн подходит, можно начать с фрагмента занятия.”
- “Для проверки мест передам менеджеру класс и предмет.”
- “Подскажите, очно или онлайн удобнее?”

Не должно быть:

- 4-5 вопросов подряд;
- “менеджер свяжется” без пользы;
- давления на продажу;
- обещания записи/места/скидки без факта.

Коды:

- `missing_next_step`
- `too_many_questions`
- `pushy_sales_tone`

### 7.5. Шаблонность и канцелярит

Ловить:

- повтор одного safe template на нескольких ходах;
- “стоимость зависит от класса, формата и периода оплаты” после того, как класс/формат уже есть;
- “повторно присылать не нужно. Отвечу по сути…” без ответа по сути;
- “менеджер проверит актуальные данные” как замена ответа;
- “спасибо за обращение” в каждом ходе;
- “ваш вопрос очень важен”;
- “оптимальный образовательный продукт”.

Коды:

- `templated_opening`
- `empty_usefulness_claim`
- `generic_price_template_after_slots_known`
- `repeated_safe_template`
- `bureaucratic_tone`

### 7.6. Составной вопрос

Если клиент спросил 2+ безопасных подпункта, бот должен ответить на основные безопасные части.

Пример:

> Сколько стоит онлайн на год? Есть скидка на второй предмет? Можно частями? Занятия в прямом эфире или записи?

Нельзя отвечать только про рассрочку.

Коды:

- `missed_question_parts`
- `single_topic_answer_to_multitopic_question`

Если в составном вопросе есть P0, route должен быть `manager_only`.

## 8. Режимы переписывания

### 8.1. Deterministic rewrite для типовых safe-template ошибок

Сначала сделать быстрые детерминированные переписывания для самых частых паттернов. Это дешевле и стабильнее, чем второй LLM-вызов.

Приоритетные сценарии:

1. Цена + “на сейчас / можно зафиксировать”.
2. Рассрочка Фотон.
3. Рассрочка УНПК.
4. Пробное Фотон онлайн.
5. Пробное УНПК.
6. Лагерь: что входит.
7. Лагерь: места есть.
8. No-fact price/schedule.
9. Off-topic.
10. Прямой вопрос “вы бот?”.

### 8.2. LLM rewrite для сложных случаев

Если детерминированный rewrite не подходит, использовать второй короткий LLM-вызов.

Критически важно: LLM rewrite не запускается на каждое сообщение. Он запускается только при явном finding:

- `ignored_direct_question`;
- `answered_nearby_topic`;
- `lost_followup_context`;
- `over_handoff_with_verified_fact`;
- `generic_price_template_after_slots_known`;
- `empty_usefulness_claim`;
- `repeated_safe_template`;
- `single_topic_answer_to_multitopic_question`;
- `missed_question_parts`.

Если findings нет, первичный ответ не переписывать.

Создать prompt builder:

`build_answer_quality_rewrite_prompt(...)`

Он получает:

- исходное сообщение клиента;
- recent messages;
- текущий слабый draft;
- route/topic/risk;
- known slots;
- confirmed facts;
- missing facts;
- findings;
- few-shot correction examples;
- строгие запреты P0/brand/fact;
- требуемый формат JSON.

Ответ rewriter:

```json
{
  "draft_text": "...",
  "answer_quality_notes": ["ответил на прямой вопрос", "не спросил известное"],
  "rewrite_reason": "over_handoff_with_verified_fact"
}
```

Rewriter не выбирает route самостоятельно, кроме безопасного понижения:

- может оставить route;
- может понизить `bot_answer_self_for_pilot` до `draft_for_manager`;
- не может повысить `manager_only` до автономного ответа;
- не может снять P0-флаги.

### 8.3. Ограничение стоимости

LLM rewrite включать только если:

- `MANGO_TELEGRAM_ANSWER_REWRITER_ENABLED=1`;
- finding severity = `rewrite`;
- route не `manager_only` по P0;
- нет hard safety flags:
  - `high_risk_manager_only`;
  - `combined_high_risk_manager_only`;
  - `refund_zero_collect`;
  - `legal_threat`;
  - `payment_confirmation_blocked`;
  - `brand_separation_blocked`.

Для первых тестов можно включить rewriter в dynamic sim, но не включать в live Telegram без отдельного решения.

Ограничения производительности и fail-closed:

- deterministic assessment/rewrite включены без второго LLM-вызова;
- LLM rewrite — только по флагу окружения и только для finding severity=`rewrite`;
- timeout второго вызова не больше основного provider timeout и должен быть явно задан в коде;
- при timeout/ошибке rewriter возвращает уже guarded draft без ожидания и без ухудшения route;
- в отчётах считать долю сообщений, где потребовался rewrite, p50/p95 latency и причины fail-closed;
- целевой лимит для пилота: LLM rewrite не чаще чем в 20-30% клиентских ходов, иначе нужно чинить первый проход, а не делать второй вызов нормой.

## 9. P0 и manager_only

P0-ответы не надо “очеловечивать” сильнее. Они должны оставаться сухими:

- “Приняли обращение. Передам ответственному сотруднику…”
- без “понимаю”, “извините”, “разберёмся”, “это важно”;
- без сбора ФИО/договора/телефона/суммы;
- без продажи после P0.

`answer_quality_rewriter` для P0 должен только проверить, что P0-текст не испорчен, но не пытаться делать его теплее.

## 10. Конкретные паттерны из последнего прогона

### 10.1. Цена: “это на сейчас / можно зафиксировать?”

Проблема:

Бот назвал цену на 1-м ходе, а на 2-м сказал:

> стоимость зависит от класса, формата и периода оплаты. Менеджер проверит...

Нужно:

> Да, это текущая подтверждённая цена на сейчас. Зафиксировать оформление за вами сможет менеджер при записи; я передам ему, что вы смотрите 8 класс, физику, онлайн.

Без:

- будущей цены;
- точной даты повышения;
- обещания “закрепил”.

### 10.2. Рассрочка Фотон

Проблема:

Бот говорит 6/10/12 и Долями, но не отвечает про банк/проценты/4 платежа и иногда перескакивает к лагерям.

Нужно:

> Да, в Фотоне можно оплатить частями. Есть рассрочка на 6, 10 или 12 месяцев и Долями на 4 части. Условия по рассрочке зависят от выбранного способа и решения банка/сервиса, поэтому одобрение я не обещаю. Если хотите, менеджер поможет оформить подходящий вариант дистанционно.

### 10.3. Рассрочка УНПК

Проблема:

Факт верный, но один и тот же fallback повторяется дословно.

Нужно варьировать:

Первый ответ:

> В УНПК банковской рассрочки нет, зато можно платить помесячно, за семестр или за год. За семестр скидка 10%, за год — 14%.

Второй ответ:

> Да, банк не нужен: это не рассрочка через банк, а варианты оплаты внутри УНПК. Если хотите растянуть оплату, менеджер подскажет удобный график.

### 10.4. Пробное Фотон онлайн

Проблема:

Бот не отвечает прямо, хотя факт есть.

Нужно:

> Да, пробное занятие есть. Для онлайн-формата его можно оформить дистанционно при записи; приезжать никуда не нужно. Напишите класс и предмет — подберём подходящий вариант.

Не говорить “бесплатное” как обещание.

### 10.5. Пробное УНПК

Нужно:

> По очному формату сейчас не начинаем с бесплатного пробного занятия: менеджер расскажет про формат, преподавателей и поможет понять, подойдёт ли программа. По онлайн-формату можно прислать фрагмент занятия, чтобы посмотреть подачу и уровень.

На вопрос “фрагмент бесплатный?”:

> Фрагмент для знакомства по онлайн-формату можем прислать; условия отправки менеджер подтвердит, чтобы не назвать неверно.

### 10.6. Лагерь: что входит

Проблема:

Бот знает цену, но не отвечает про проживание/питание/трансфер.

Нужно для Фотона:

> По выездной школе в Менделеево формат с проживанием и организованной программой. По питанию и трансферу сориентируем по конкретной смене; наличие мест менеджер проверит отдельно. Сейчас по Фотону цена 93 100 ₽, полная — 98 000 ₽.

Нужно для УНПК:

> В ЛВШ Менделеево УНПК входит проживание и 5-разовое питание. Полная стоимость — 120 000 ₽, текущая — 114 000 ₽. Наличие мест сейчас проверяет менеджер.

Если класс 11:

> Важный момент: сама ЛВШ обычно для окончивших 5-10 класс, поэтому для 11 класса менеджер проверит подходящую альтернативу.

### 10.7. Места есть?

Нельзя:

- “места есть”;
- “забронирую”;
- “закреплю место”.

Нужно:

> По местам не буду обещать без проверки. Класс и предмет уже вижу: передам менеджеру, чтобы он проверил наличие по конкретной смене.

## 11. Тесты

### 11.1. Unit tests нового модуля

Создать:

`tests/test_answer_quality_rewriter.py`

Покрыть:

- `ignored_direct_question`;
- `over_handoff_with_verified_fact`;
- `reasked_known_grade`;
- `generic_price_template_after_slots_known`;
- `single_topic_answer_to_multitopic_question`;
- P0 не переписывается в тёплый sales-текст;
- brand leak после rewrite ловится финальным guard;
- unsupported numbers после rewrite ловятся финальным guard.

### 11.2. Provider integration tests

Обновить:

- `tests/test_subscription_llm_draft_provider.py`;
- `tests/test_draft_prompt_builder.py`;
- `tests/test_telegram_dynamic_client_sim.py`;
- `tests/test_telegram_few_shot_reference.py`.

Проверить:

- rewriter вызывается после первичных guards и до финального autonomy guard;
- final guards выполняются после rewrite;
- `safety_flags` включает `answer_quality_rewritten` при переписывании;
- `metadata.answer_quality` содержит findings;
- `answer_quality_notes` не теряются.

### 11.3. Regression tests из targeted16

Добавить frozen cases:

1. `v8_foton_t01_pricing_01`: второй ход “цена на сейчас / закрепится?”.
2. `v8_foton_t02_installment_02`: “помесячно или банк? Долями на 4 платежа?”.
3. `v8_foton_t03_trial_02`: “онлайн точно, никуда приезжать не надо?”.
4. `v8_unpk_t02_installment_01`: второй ответ не должен дословно повторять первый.
5. `v8_unpk_t10_camp_02`: “проживание и питание? полная стоимость 114 или 120?”.
6. `batch_foton_01`: составной вопрос онлайн цена + скидка + рассрочка + записи.

### 11.4. Context parity tests

Добавить тесты, которые доказывают, что live Telegram runtime и dynamic sim передают в prompt один и тот же класс смысловых полей:

- `recent_messages`;
- `known_dialog_fields`;
- `known_slots`;
- `funnel_state`;
- `answer_quality_reference`;
- `few_shot_style_examples`;
- `few_shot_correction_examples`.

Если targeted16 запускается без `funnel_state/known_slots`, результат должен быть помечен как `context_parity_incomplete`.

### 11.5. v6/v5 regrading tests

Перед тем как считать v6/v5 FAIL реальными ошибками, нужно перегрейдить ожидания:

- полезный честный автономный ответ без точных чисел может быть допустимым, если нет P0 и нет выдумки;
- `draft_for_manager` и `manager_only` могут быть более осторожным маршрутом, если клиент спрашивает будущую цену или точное расписание без факта;
- route mismatch не считается safety FAIL без forbidden hits, P0 leakage или unsupported promise.

Нужен отчёт:

- `real_safety_fail`;
- `route_expectation_outdated`;
- `answer_incomplete`;
- `test_marker_too_strict`.

### 11.6. P0/no-fact/price-date regression gate

Отдельно от общего v6/v5 счётчика проверить:

- P0 не переписывается;
- no-fact ответы не содержат новых цен, дат, процентов, обещаний мест или фиксации;
- price-date кейсы не называют будущие цены и не обещают срок действия цены без факта;
- составные вопросы с P0 остаются `manager_only`, даже если безопасные подпункты можно было бы ответить.

## 12. Метрики успеха

После реализации повторить `v8_targeted16`.

Acceptance для первого релиза:

- hard FAIL: 0;
- `ignored_question`: не больше 4/16;
- `over_handoff`: не больше 4/16;
- `templated_opening`: не больше 5/16;
- средний human tone: минимум 62/100;
- ни одного P0-regression;
- ни одной бренд-утечки;
- ни одной неподтверждённой цены/даты/места.

Stretch target:

- хотя бы 8/16 `PASS`;
- средний human tone 70+;
- `PASS_WITH_NOTES` остаётся только для спорных бизнес-фактов, а не для шаблонности.

## 13. Порядок реализации

### Фаза 0. Preflight

- Проверить `git status`.
- Не трогать unrelated изменения.
- Зафиксировать текущие результаты:
  - `audits/_inbox/telegram_dynamic_client_sim_v8_targeted16_20260523_145732/`;
  - `audits/_inbox/telegram_static_v6_v5_quick_20260523_152353/`;
  - `audits/_inbox/telegram_targeted16_static_v6_v5_review_20260523/`.

Дополнительно:

- проверить, что активный snapshot — `product_data/knowledge_base/kb_release_20260520_v6_3_team_answers/kb_release_v3_snapshot.json`;
- проверить, что bot-pack и generator синхронизированы после policy C;
- проверить, что targeted16 запускается против текущей рабочей папки, а не старой копии.
- подтвердить, что `product_data/knowledge_base/kb_release_20260520_v6_3_team_answers_bot_pack/manifest.json` и активный snapshot указывают на актуальную v6.3 после последних правок;
- проверить, что rev2 corpus Claude синхронизирован с рабочими YAML/JSON, если эти файлы используются в prompt/reference.

### Фаза 0.5. Context parity: live Telegram vs dynamic sim

До rewriter:

- сравнить `scripts/run_telegram_public_pilot_bots.py` и `scripts/run_telegram_dynamic_client_sim.py`;
- добавить в dynamic sim недостающие поля контекста, если live их уже использует;
- гарантировать, что dynamic sim строит `funnel_state` и `known_slots` тем же способом, что live Telegram;
- гарантировать, что few-shot и answer-quality references реально попадают в prompt в targeted16.

Acceptance Фазы 0.5:

- тест показывает, что dynamic context содержит `funnel_state`, `known_slots`, `few_shot_style_examples`, `few_shot_correction_examples`;
- targeted16 summary содержит отметку `context_parity_checked: true`.

### Фаза 0.7. Усиление первого прохода до rewriter

До переписывания ответов усилить первичную генерацию:

- добавить/уточнить prompt-инструкцию: “последнее сообщение клиента важнее общего шаблона”;
- явно требовать сначала ответить на последний прямой вопрос;
- если клиент уже дал класс/предмет/формат, запретить общий шаблон “стоимость зависит от класса/формата”;
- если вопрос составной, требовать `question_parts` и краткий ответ на каждую безопасную часть;
- если нет точного факта, давать полезный частичный ответ без числа, а не пустой handoff;
- few-shot examples использовать именно для второго хода и исправления шаблонности.

Acceptance Фазы 0.7:

- unit tests доказывают, что prompt содержит answer-first и multitopic-first правила;
- targeted16 после этой фазы можно сравнить как “первый проход улучшен без rewrite”.

### Фаза 1. Анализатор качества без rewrite

Реализовать `assess_answer_quality`.

Подключить в provider только для metadata/safety_flags, без изменения текста. Это shadow-режим: он не должен менять клиентский ответ, но должен записывать, что именно было бы переписано.

Прогнать unit tests.

Цель: понять, что detector ловит те же проблемы, что судья targeted16.

### Фаза 2. Детерминированный rewrite для top-7 паттернов

Реализовать безопасные переписывания:

- price follow-up;
- Foton installment;
- UNPK installment second turn;
- Foton online trial;
- UNPK trial;
- camp included;
- seats availability.

Rewrite запускается только по findings из Фазы 1. Если finding нет, ответ не трогать.

После deterministic rewrite сразу повторять hard guards.

### Фаза 3. LLM rewrite за feature flag

Добавить второй короткий LLM-вызов для сложных случаев.

Флаг:

`MANGO_TELEGRAM_ANSWER_REWRITER_ENABLED=1`

По умолчанию в live Telegram оставить выключенным до проверки.

### Фаза 4. Финальные guards после rewrite

После rewrite обязательно повторить:

- brand guard;
- input/high-risk guard;
- unsupported promise guard;
- operational specificity guard;
- known context redundant question guard;
- funnel guard;
- autonomy matrix guard.

### Фаза 5. Targeted16 повтор

Прогнать `v8_targeted16`.

Сравнить “было → стало”:

- hard FAIL;
- soft flags;
- tone score;
- PASS count;
- конкретные 16 диалогов.

### Фаза 6. v6/v5 повтор

Прогнать статичные `v6/v5`.

Отдельно отметить:

- реальные safety FAIL;
- устаревшие route expectations;
- смысловые ответы, где route допустим, но текст неполный.

Не считать общий счётчик FAIL финальным без regrading. Для решения использовать semantic buckets:

- `real_safety_fail`;
- `answer_incomplete`;
- `route_expectation_outdated`;
- `test_marker_too_strict`.

### Фаза 7. Решение по live Telegram

Если targeted16 проходит acceptance:

- включить deterministic rewrite сначала только в shadow/journal-only или во внутреннем тестовом режиме;
- LLM rewrite оставить выключенным или включить только для внутренних тестов;
- затем per-brand canary только после отдельного подтверждения Дмитрия;
- обновить RUNBOOK;
- перезапускать live/poll ботов только после отдельного подтверждения Дмитрия;
- собрать дневной отчёт пилота.

## 14. Audit pack

После реализации создать:

`audits/_inbox/answer_quality_rewriter_<timestamp>/`

Обязательные файлы:

- `implementation_notes.md`;
- `changed_files.txt`;
- `test_output.txt`;
- `semantic_review.md`;
- `risk_review.md`;
- `before_after_targeted16.md`;
- `static_v6_v5_review.md`;
- `known_limitations.md`.

## 15. Критерии “готово”

Блок считается выполненным, если:

1. Новый модуль реализован и покрыт unit/integration tests.
2. P0/brand/fact guards после rewrite работают.
3. `v8_targeted16` улучшился по soft-флагам и tone.
4. Нет новых hard FAIL.
5. Составлен diff “было → стало”.
6. Есть semantic review `PASS` или `PASS_WITH_NOTES` без блокеров для внутреннего пилота.

## 16. Риски

### Риск 1. Rewriter начнёт выдумывать

Митигировать финальными guards и запретом использовать few-shot как источник фактов.

### Риск 2. Rewriter ослабит P0

P0 не переписывать, кроме возможного сухого safe-template. Manager-only не повышать.

### Риск 3. Станет медленнее

Детерминированный rewrite делать первым. LLM rewrite — feature flag и только для проблемных случаев.

### Риск 4. Подгонимся под targeted16

Использовать targeted16 как быстрый acceptance, но не считать его финалом. После него нужен v6/v5, затем полный v8 и живой пилот.

### Риск 5. Дублирование funnel-слоя

Rewriter не должен заново строить воронку. Он использует уже готовые `known_slots`, `next_best_question`, `funnel_state` и исправляет только качество текста.

## 17. Следующие шаги после этого ТЗ

1. Согласовать ТЗ с Дмитрием.
2. Реализовать Фазы 0-2.
3. Прогнать unit/integration tests.
4. Прогнать `v8_targeted16`.
5. Если улучшение есть и hard FAIL = 0 — реализовать Фазу 3 под feature flag.
6. Повторить targeted16 + v6/v5.
7. Только после этого решать, идти ли к полному v8.
