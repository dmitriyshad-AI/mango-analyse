# ADR-003 Regex Understanding Moratorium

Дата: 2026-07-01

## Правило

Смысл клиентского сообщения в direct-path должен определять SemanticFrame/LLM, а не новый regex или keyword-детектор.

Новый клиентский смысловой сбой проходит через такой путь:

1. добавить пример в eval-набор;
2. улучшить инструкцию/схему/калибровку SemanticFrame;
3. проверить метриками и сырьем;
4. только потом включать поведение за флагом.

Нельзя добавлять новый `re.compile` или keyword-таблицу, которая читает сырой текст клиента и решает:

- P0/не P0;
- intent/topic;
- close/not-close;
- requested action;
- answerability/relevance;
- venue/scope/product meaning;
- готовность к оплате или сделке.

## Что остается разрешенным детерминизмом

Разрешены механические проверки выхода и инфраструктурные парсеры:

- ПДн/телефон/email/id scrub;
- проверка чисел, дат, ссылок, брендов и обещаний в тексте ответа;
- fail-closed, когда модель не ответила или дала низкую уверенность;
- тестовые/отчетные парсеры.

Любое расширение regex в runtime-канале должно явно объяснить, что это проверка выхода, а не понимание клиента, и обновить guard-тест.

## CI guard

Мораторий закреплен тестом `tests/test_adr003_regex_understanding_moratorium.py`.

Он проверяет два frozen snapshot:

- `tests/fixtures/adr003_runtime_channel_regex_snapshot.json` — все текущие `re.compile` в runtime-каналах;
- `tests/fixtures/adr003_direct_path_text_patterns_snapshot.json` — direct-path `re.compile`, inline `re.search/sub/...`, верхнеуровневые keyword/marker таблицы и строковые `"..." in text`-проверки в файлах понимания.

Если тест упал, нельзя просто обновить snapshot. Сначала надо ответить:

1. это проверка выхода/fail-closed/PII/brand/fabrication, а не понимание сырого клиентского текста?
2. есть ли eval-кейс, который фиксирует найденный смысловой сбой?
3. почему это не должно решаться SemanticFrame?

Если это новый смысл клиента, snapshot не обновляется: добавляется eval-кейс и калибруется SemanticFrame. Если это разрешенная механическая проверка выхода, в audit pack нужно явно написать причину и только затем обновить snapshot.

## Разрешенное обновление 2026-07-01: SemanticFrame manager-action gate

Snapshot `tests/fixtures/adr003_direct_path_text_patterns_snapshot.json` обновлен из-за новых технических констант:

- `TELEGRAM_SEMANTIC_FRAME_MANAGER_ACTION_GATE`;
- schema/version и порог confidence;
- закрытые enum-наборы `deal_stage`/`payment_readiness`, которые читают уже готовый `semantic_frame`.

Это не новое regex/keyword-понимание сырого клиентского текста. Гейт не парсит сообщение клиента и не добавляет словари маркеров. Он работает только по posthoc `semantic_frame` со статусом `ok`, за default-OFF флагом, и может только повысить автономный маршрут до `draft_for_manager`; текст ответа не меняет.

## Разрешенное обновление 2026-07-03: Э0 semantic-reading foundation

Snapshot `tests/fixtures/adr003_runtime_channel_regex_snapshot.json` обновлен из-за удаления 9 мертвых regex-объявлений из `post_layers.py`:

- `HIGH_RISK_INPUT_PATTERNS` (4 паттерна);
- `LEGAL_CONTEXT_INPUT_RE`;
- `ZERO_COLLECT_DRAFT_RE`;
- `REFUND_FORBIDDEN_DETAIL_RE`;
- `COMPLAINT_APOLOGY_RE`;
- `COMPLAINT_DETAIL_COLLECT_RE`.

Это сужение legacy-regex бюджета: `CHANNEL_REGEX_BUDGET["src/mango_mvp/channels/subscription_llm_parts/post_layers.py"]` понижен с 73 до 64. По `rg` эти имена были только объявлениями, пассивными импортами/exports и строками fixture-снапшотов; рабочих `.search`/вызовов не было.

Snapshot `tests/fixtures/adr003_direct_path_text_patterns_snapshot.json` также расширен, потому что guard теперь покрывает:

- `src/mango_mvp/channels/text_signals.py`;
- `src/mango_mvp/channels/actions.py`;
- `src/mango_mvp/channels/answer_quality_rewriter.py`;
- `src/mango_mvp/channels/dialogue_memory.py`;
- `src/mango_mvp/channels/held_state.py`;
- `src/mango_mvp/channels/new_lead_funnel.py`;
- `src/mango_mvp/channels/subscription_llm_parts/policy_routing.py`;
- literal вызовы `has_marker` / `has_any_marker` / `has_word_marker` / `has_exact_word` и их локальные `_...`-алиасы;
- именованные верхнеуровневые `*_MARKERS` / `*_NEIGHBORS` / keyword-таблицы в перечисленных runtime-файлах.

Это не добавляет нового понимания. Наоборот, фиксирует текущий marker-helper долг как потолок (`CHANNEL_MARKER_HELPER_BUDGET`, сейчас 327 вызовов по runtime channels) и закрывает дыру, где новые marker-таблицы могли появляться без SemanticFrame/eval.

Отдельно: commit-level pre-push/CI guard из Foton-ТЗ в этом заходе не включен как реальный hook/workflow. Причина — это изменение общего процесса пуша, а не runtime-кода; его нужно делать отдельным ops-ТЗ. Локально мораторий обеспечивается pytest-guard'ом и frozen snapshots.

В этом же обновлении snapshot фиксирует добавление `off_topic` в закрытый enum `model_intent`. Это не regex/keyword-понимание: значение приходит из LLM-блока `model_intent`, нужно для сохранения всего блока парсером и пока не является новой маршрутной целью.

## Разрешенное обновление 2026-07-04: Ш4 первый срез legacy-понимания

После E3-регрейда и явного “да №2” удалены:

- `OFF_TOPIC_INPUT_RE` из `policy_routing.py`; off-topic terminal template теперь опирается на taxonomy `service:S3_out_of_scope` и `off_topic` semantic-reading decision;
- входной `availability` facet из `reliable_answerer.py`; понимание клиентского вопроса про места теперь относится к `sense_seats`/SemanticFrame, а не к facet regex.

Бюджет `CHANNEL_REGEX_BUDGET["src/mango_mvp/channels/subscription_llm_parts/policy_routing.py"]` понижен с 16 до 15. Frozen snapshots обновлены намеренно.

Что оставлено осознанно:

- `_AVAILABILITY_PROMISE_RE` и `availability_promise_detected()` в `reliable_answerer.py` остаются. Это проверка уже сгенерированного ответа на опасное обещание мест/группы/брони без live-факта, а не понимание сырого клиентского текста.
- `dialogue_memory.py` grade/subject/format extraction пока остаётся. На текущем HEAD `semantic_reading_slots` являются hidden storage и не заменяют `known_slots`; удалять старое извлечение можно только после отдельного `slots_gsf -> known_slots` merge с `source=semantic_reading_llm` и запретом попадания в `client_confirmed_slots`.

## Разрешенное обновление 2026-07-04: PR-A Fix1b verified-fact autonomy corridor

Добавлен default-OFF флаг `TELEGRAM_FIX1B_AUTONOMY_VERIFIED_FACTS`.

Это не новый детектор смысла клиента. Флаг не читает сырой клиентский текст для выбора intent/topic/P0. Он только сужает ложные выходные демоуты `autonomy_default_cautious_missing_facts` и `autonomy_default_cautious_unverified_fact`, когда уже готовый черновик:

- относится к разрешенной autonomy-теме;
- имеет определенный бренд;
- не P0/high-risk;
- полностью поддержан свежими client-safe фактами;
- не содержит неподтвержденных чисел/дат/брендов;
- не обещает live-наличие мест/групп/брони.

Live-status пол, P0-пол, brand/fabrication проверки и legacy availability-promise floor не обходятся. Новые стоп-юниты покрывают частичную поддержку: лишнее число, чужой бренд и живые места.

## Разрешенное обновление 2026-07-04: PR-D rolling dialog summary

Добавлен default-OFF флаг `TELEGRAM_DIALOG_SUMMARY_ROLLING`.

Это не новый regex-понимальщик клиента. Сводка производится моделью в уже существующем direct-path вызове через аддитивное поле `dialog_summary` и не добавляет отдельный LLM-вызов. Детерминированная часть только верифицирует запись в память: отбрасывает суммы/проценты/даты через существующий summary-фильтр, ПДн через существующие phone/email regex из direct-path support и чужой бренд через brand-token guard без новых `re.compile`.

Сводка не пишет `known_slots`, `client_confirmed_slots`, CRM/Tallanto/AMO и не включает флаг в профиль. При OFF старый slot-glue `conversation_summary_short` сохраняется.

Snapshot `tests/fixtures/adr003_direct_path_text_patterns_snapshot.json` обновлен намеренно на две технические проверки в provider: наличие `"dialog_summary"` и маркера `ПРЕДЫДУЩАЯ СВОДКА` в уже собранном prompt. Это не чтение смысла клиентского текста, а выбор, нужно ли нормализовать аддитивное JSON-поле из того же LLM-вызова.

После аудита snapshot расширен ещё одной fail-closed проверкой в `dialogue_memory.py`: строка `процент` внутри `_summary_has_unsupported_number`. Это не классификация клиентского запроса; это запрет записи model-generated rolling summary в память, если модель внесла процент/скидку без проверенного факта.

## Разрешенное обновление 2026-07-04: Package-2 Srez-1a intent_actions shadow

Добавлен reading-класс `intent_actions` в allowlist `TELEGRAM_SEMANTIC_READING_CLASSES`.

Это не новый regex/keyword-понимальщик клиента и не профильное включение. Класс работает только при явном env в измерительной ON-ноге и читает только inline `SemanticFrame` из уже готового direct-path JSON на той же стадии, где раньше применялся `conversation_intent_plan` output guard.

В переходном режиме legacy-guard считается как раньше, а frame-решение может только:

- fail-close к legacy при отсутствии inline frame, низкой уверенности, невалидном enum или posthoc/source mismatch;
- переустановить защитный сигнал `conversation_intent_plan_live_availability` для `requested_action=check_availability`;
- записать trace конфликта для регрейда.

Снятие ложного P0 в этом коммите не переносится на frame: legacy-поведение остаётся единственным источником такого repair до отдельного замера.

`intent_actions` не добавлен в `PILOT_PROFILE_DEFAULT_READING_CLASSES`; профильное включение возможно только отдельным решением после M1-пары и регрейда.

Обновление 2026-07-05 после свежего регрейда пары 1a и отдельного «да» владельца: это отдельное решение принято. `intent_actions` теперь входит в профильный default `pilot_gold_v1`, а legacy output-ветка `primary_intent == "live_availability"` в `conversation_intent_plan` guard удалена. Входной детектор live_availability и `live_status_read/conversation_intent_plan` apply-point этим шагом не включались и не удалялись. Старый plan live_availability остаётся только как fail-closed fallback при отсутствующем/невалидном inline-frame, не как штатный пониматель.

## Разрешенное обновление 2026-07-04: профиль PaymentFix/PR-D и подготовка среза-1a

После M1-регрейда PaymentFix и PR-D переведены в default-ON профиль `pilot_gold_v1` с сохранением явного `=0` override:

- `TELEGRAM_TEXT_HYGIENE_PAYMENT_FIX` — текстовая гигиена оплаты/возврата, не новый regex-понимальщик клиента;
- `TELEGRAM_DIALOG_SUMMARY_ROLLING` — запись безопасной сводки из inline LLM-поля `dialog_summary`, без записи в `known_slots`/CRM/Tallanto/AMO.

`TELEGRAM_FIX1B_AUTONOMY_VERIFIED_FACTS` остается default-OFF. Коридор ужесточен до включения: отрицательные утверждения о существовании и paid-context во входе не промоутятся.

Runner для Package-2 получает только измерительный `TARGET_READING_CLASS`: `intent_actions` добавляется в ON-ногу через env, но не в профильный default. Это сохраняет чистый baseline.

Уточнение после регрейда среза-1a: B-нога runner больше не задаёт пустой `TELEGRAM_SEMANTIC_READING_CLASSES=` и должна запускаться как чистый профиль `pilot_gold_v1`; ON-нога добавляет target-class явным env. Пара `72c84090` полезна как разведка, но не является финальным основанием для deletion legacy live-availability ветки. Deletion требует свежей пары после фикса runner и отдельного решения.

Inline text health gate получил узкую верификацию адресных чисел `20`/`30` и учебного года `2026/27` из selected exact fact текущего хода: id, текста или metadata выбранного exact-факта. Raw fact blob и adjacent facts не становятся pass-источником; служебные даты вида `2026_06_11` не считаются учебным годом.

## Разрешенное обновление 2026-07-07: child-safety P0 floor перед swap

Расширен детерминированный P0 safety floor для сигналов:

- ребёнок один остался после занятия;
- ребёнка никто не встретил;
- после занятия оставили одного;
- ребёнок без присмотра/надзора.

Это не новый слой бизнес-понимания для ответа клиенту. Правило не выбирает продукт, тему, цену, действие или автономность. Оно только fail-close блокирует прямой путь до модели и отдаёт короткую manager-only передачу без сбора деталей и без справочных фактов. Цель — не зависеть от классификации модели в child-safety случаях перед live swap.

Негативные контроли добавлены на обычные вопросы про расписание, выбор преподавателя и порядок встречи у кабинета: они не должны становиться P0 только из-за слова «ребёнок» или «встречать».

Snapshot `tests/fixtures/adr003_runtime_channel_regex_snapshot.json` и `tests/fixtures/adr003_direct_path_text_patterns_snapshot.json` обновлены намеренно. Бюджет regex не повышался: изменены существующие P0/backstop regex в `p0_recall_spec.py` и `support.py`.
