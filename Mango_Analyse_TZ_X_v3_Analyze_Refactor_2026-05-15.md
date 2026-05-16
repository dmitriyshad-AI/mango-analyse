# ТЗ-X-v3: финальное ТЗ на рефакторинг analyze.py и честное отключение resolve LLM

Дата: 2026-05-15
Адресат: Codex как исполнитель реализации
Основа: `Mango_Analyse_TZ_X_v2_Analyze_Refactor_2026-05-14.md` и `Mango_Analyse_TZ_X_v2.1_Delta_2026-05-15.md`

## Контекст и проблема

Цель X-трека - привести входной анализ звонков к более надежному состоянию без переписывания всего слоя. Сейчас основные риски сосредоточены в четырех местах.

Первое: full-профиль анализа получает более длинный текст, но не получает те же детерминированные подсказки, которые получает compact-профиль. В `src/mango_mvp/services/analyze.py:787-825` функция `_analysis_prompt_context()` возвращает словарь с ключами `profile`, `system_prompt`, `user_prompt`, `llm_prompt`, `metrics`. Подсказки добавляются только при `normalized == "compact"` на `analyze.py:811-816`. При этом `SYSTEM_PROMPT_FULL` на `analyze.py:26-56` говорит использовать только транскрипт и метаданные, а `SYSTEM_PROMPT_COMPACT` на `analyze.py:58-89` уже учитывает deterministic hints. Это создает разный режим мышления модели.

Второе: `_compose_history_summary()` на `analyze.py:1342-1484` не добавляет в историю клиента часть уже извлеченных полезных полей: школу, коммерческий контекст, приоритет лида. Функция имеет две ветки: ветка с LLM-черновиком на `analyze.py:1399-1452` и fallback-ветка без черновика на `analyze.py:1454-1484`. Правка должна попасть в обе ветки, иначе поведение будет зависеть от того, вернула ли модель хороший `history_summary`.

Третье: дедупликация коротких filler-реплик может выкинуть важные подтверждения клиента. Сейчас `_filler_only_signature()` на `analyze.py:608-615` возвращает сигнатуру для любых строк, состоящих только из filler-токенов. В `_compact_transcript_for_prompt()` на `analyze.py:671-679` повтор с тем же спикером удаляется. Для двух подряд строк `Клиент: Да.` это опасно: "да" может быть согласием, а не мусором.

Четвертое: старый план X-B предлагал escape hatch, который мог сохранить выдуманные продажные поля модели и перевести звонок из `non_conversation` в продажный тип. Это рискованно. Если модель на автоответчике или системной фразе выдумала продукт, следующий шаг или возражение, такой escape hatch превратит мусор в действие для менеджера. Поэтому в X-v3 выбран вариант **soft warning**: не переопределять `call_type`, не сохранять LLM-поля как факт, а только поставить флаг ручной проверки в `quality_flags` и `review_reasons`.

Отдельно X-D: `resolve.py` сейчас считает `llm_used` нечестно при `RESOLVE_LLM_PROVIDER=off`. В `src/mango_mvp/services/resolve.py:1412-1533` `_resolve_with_llm()` при выключенном провайдере не выходит сразу, а меняет провайдера на `"rule"` и все равно возвращает кандидата с `name="llm"` на `resolve.py:1480` или `resolve.py:1520`. Затем `llm_used` увеличивается на `resolve.py:1805`. Это создает ложную метрику использования LLM.

Финальный порядок реализации: **X-A -> X-D -> X-B -> X-C**. X-C в этой версии не включается в основной пакет реализации, а фиксируется как будущий эксперимент за отдельным флагом. Причина: X-C меняет стратегию обрезки длинных транскриптов и требует отдельного набора длинных тестовых звонков; сейчас быстрее и безопаснее закрыть локальные ошибки X-A, честную метрику X-D и мягкий контроль X-B.

## X-A. Конкретные правки

### X-A.1. Full-профиль получает deterministic hints

Файл: `src/mango_mvp/services/analyze.py:26-56`, `src/mango_mvp/services/analyze.py:58-89`, `src/mango_mvp/services/analyze.py:161-164`, `src/mango_mvp/services/analyze.py:787-825`.

Нужно изменить условие на `analyze.py:811` с `if normalized == "compact":` на `if normalized in {"compact", "full"}:`. Подсказки должны попадать в `payload["user_prompt"]`, не в несуществующий ключ `payload["prompt"]`.

В `SYSTEM_PROMPT_FULL` на `analyze.py:30` заменить смысл правила с "Use only transcript + metadata facts" на "Use only transcript + metadata + deterministic hints when supported by transcript facts". Формулировка должна прямо запрещать выдумывать факты из подсказок.

Так как меняется системный промпт, поднять `ANALYZE_PROMPT_VERSION_FULL` на `analyze.py:163` с `"v6"` до `"v7"`. `ANALYZE_PROMPT_VERSION_COMPACT` на `analyze.py:162` не менять, если compact prompt не меняется.

### X-A.2. История клиента дополняется школой, коммерческим контекстом и приоритетом

Файл: `src/mango_mvp/services/analyze.py:1342-1484`, источники полей на `analyze.py:1626-1651` и `analyze.py:1703-1707`.

Добавить три маленьких helper-метода рядом с `_compose_history_summary()`:

```python
def _build_commercial_lines(self, structured_fields: Dict[str, Any]) -> list[str]:
    commercial = self._nested_dict(structured_fields, "commercial")
    ...

def _build_school_line(self, structured_fields: Dict[str, Any]) -> Optional[str]:
    student = self._nested_dict(structured_fields, "student")
    ...

def _build_lead_priority_line(self, structured_fields: Dict[str, Any]) -> Optional[str]:
    ...
```

Точные значения из текущего кода:

- `commercial.price_sensitivity`: `"high"`, `"medium"`, `"low"` на `analyze.py:1639-1645`. В историю писать как "высокая", "средняя", "низкая".
- `commercial.budget`: строка из `commercial.budget` или legacy `raw["budget"]` на `analyze.py:1629`; пустые и "не указан" не писать.
- `commercial.discount_interest`: boolean на `analyze.py:1647-1651`; писать только если `True`.
- `lead_priority`: `"hot"`, `"warm"`, `"cold"` на `analyze.py:1703-1707`; писать только `"hot"` и `"warm"`, потому что `"cold"` будет шумом.
- `student.school`: `student.school` на `analyze.py:1626`; писать только если непусто.

В ветке с черновиком `analyze.py:1399-1452` добавлять строки с защитой от дублей через `_summary_mentions_any()` на `analyze.py:1332-1340`. Проверять не весь список `parts`, а текущий текст черновика/уже собранные части. В fallback-ветке `analyze.py:1454-1484` добавлять без проверки дублей, потому что черновика нет.

Важная техническая деталь: `_summary_mentions_any()` на `analyze.py:1332-1340` принимает строку `text`, а не список. Нельзя передавать туда `parts` как список. Если нужно проверить уже собранные части, сначала собрать строку `" ".join(parts)`, либо проверять исходный `compact_draft`. Это отдельный источник прошлых ошибок в ТЗ X-v2.1.

Нужно избегать длинных абзацев внутри history. Все добавленные строки должны быть короткими и сканируемыми: `Коммерческий контекст: чувствительность к цене: высокая; бюджет: 50000; интересуется скидками.`, `Школа: школа №16.`, `Приоритет лида: горячий.` Формулировки могут быть чуть другими, но они не должны добавлять новых обещаний клиенту. Это не текст для отправки клиенту, а заметка для CRM/менеджера.

### X-A.3. Не дедуплицировать важные "да" и "спасибо"

Файл: `src/mango_mvp/services/analyze.py:593-615`, `src/mango_mvp/services/analyze.py:626-720`, существующий тест на `tests/test_analyze.py:457-476`.

Добавить рядом с filler-токенами набор:

```python
PROMPT_COMPACTION_COMMITMENT_TOKENS = {"да", "спасибо"}
```

В `_filler_only_signature()` вернуть `None`, если:

- токенов нет;
- не все токены входят в `PROMPT_COMPACTION_FILLER_TOKENS`;
- токен один;
- среди токенов есть `"да"` или `"спасибо"`.

Не использовать `id(text)`. В текущем условии дедупа на `analyze.py:672-679` значение `None` означает, что межстрочная дедупликация не сработает. Внутристрочная чистка `"Да, да, да"` через `_compact_prompt_filler_body()` на `analyze.py:594-605` должна остаться.

Существующий тест `tests/test_analyze.py:457-476` сейчас ожидает, что два подряд `Клиент: Да` сожмутся до одного. Этот assert нужно инвертировать или заменить новым тестом: два отдельных "Да" от клиента должны сохраниться.

## X-D. Конкретные правки

Файлы: `src/mango_mvp/services/resolve.py:1412-1533`, `src/mango_mvp/services/resolve.py:1788-1805`, `src/mango_mvp/config.py:50-115`, `src/mango_mvp/config.py:152-242`, `.env.example`.

В начало `_resolve_with_llm()` после `llm_provider = ...` на `resolve.py:1422` добавить early return:

```python
if llm_provider not in {"ollama", "openai", "codex_cli"}:
    return None
```

Не переименовывать `name`, не создавать кандидата `"llm"` при выключенном провайдере. Тогда `llm_used += 1` на `resolve.py:1805` не будет выполняться, потому что caller не получит `llm_candidate`.

В `src/mango_mvp/config.py` `Settings` - это `@dataclass(frozen=True)`, не Pydantic. Поэтому менять нужно dataclass-поле и `get_settings()`. На `config.py:236` заменить default `RESOLVE_LLM_PROVIDER` с `"codex_cli"` на `"off"`.

В `.env.example` явно указать:

```text
RESOLVE_LLM_PROVIDER=off
```

Создать документ `docs/RESOLVE_LLM_DISABLED_2026-05-15.md`: почему выключено, как включить обратно, почему метрика `llm_used` после правки означает реальные LLM-вызовы.

## X-B. Конкретные правки: soft warning вместо escape hatch

Файлы: `src/mango_mvp/quality/non_conversation.py:21-59`, `src/mango_mvp/quality/non_conversation.py:239-360`, `src/mango_mvp/services/analyze.py:91-118`, `src/mango_mvp/services/analyze.py:1556-1901`.

Выбран вариант **soft warning**. Мы не выкидываем X-B, потому что риск ложного `non_conversation` реален: compliance-преамбула может быть в живом продажном звонке. Но мы не делаем прежний escape hatch, потому что он может сохранить галлюцинации модели и привести к неправильной записи в CRM.

Нужно сделать два уровня защиты.

Первый уровень - уменьшить ложные срабатывания до LLM. В `non_conversation.py` вынести compliance-фразы из `SYSTEM_NO_DIALOGUE_RE`:

- `вас приветствует компан`
- `все разговоры записываются`
- `ваш звонок очень важен`
- `звонок может быть записан`

Создать `COMPLIANCE_PREAMBLE_RE` и не включать его в `NO_LIVE_RE`. Эти фразы сами по себе не должны снижать звонок до `non_conversation`, если есть живой диалог и учебно-коммерческий контекст.

Для `THIRD_PARTY_IVR_RE` на `non_conversation.py:45-59` добавить обход по живому учебному контексту рядом с текущим `live_payment_context` на `non_conversation.py:276-283`. Новый helper должен использовать уже существующие `CLIENT_HUMAN_RESPONSE_RE` на `non_conversation.py:195-199` и `BUSINESS_TERM_RE` на `non_conversation.py:157-162`.

В `STRONG_NON_CONVERSATION_MARKERS` на `analyze.py:91-118` убрать compliance-дубликаты `"вас приветствует компания"` и `"все разговоры записываются"`. Оставить реальные no-live маркеры: голосовой ассистент, секретарь, абонент недоступен, нажмите 1, коллекторские организации и похожее.

Второй уровень - мягкий флаг после LLM. В `_normalize_analysis()` на `analyze.py:1556-1901` перед обнулением полей на `analyze.py:1720-1743` сохранить только факт наличия LLM-сигналов, но не сами поля как бизнес-истину:

- были ли непустые `interests.products`;
- был ли непустой `next_step.action`;
- были ли непустые `objections`;
- был ли непустой `target_product`.

Если `call_type == "non_conversation"` и такие сигналы есть, после создания `quality_flags` на `analyze.py:1758-1763` добавить:

- `quality_flags["non_conversation_soft_warning_llm_sales_signal"] = True`;
- `quality_flags["non_conversation_soft_warning_sources"] = [...]`;
- `quality_flags["needs_review"] = True`;
- в `review_reasons` добавить `"non_conversation_llm_sales_signal_soft_warning"`.

Текущий normalized payload не содержит отдельного top-level `call_type`. Тип звонка хранится в `quality_flags["call_type"]` на `analyze.py:1762`. Поэтому любые тесты и downstream-проверки должны смотреть именно `result["quality_flags"]["call_type"]`, а не `result["call_type"]`.

Soft warning должен пережить `_apply_non_conversation_hard_validation()`. Эта функция получает `normalized` на `analyze.py:1205-1209`, берет `quality_flags` на `analyze.py:1210-1212`, при `call_type == "non_conversation"` очищает structured fields и обновляет payload на `analyze.py:1263-1283`. Поэтому флаг нужно положить в `quality_flags` до вызова hard validation; тогда он останется в итоговом результате вместе с `non_conversation_hard_validation_applied`.

Top-level `needs_review` и `review_reasons` тоже можно дополнить, но это вторично. Главный контракт X-B - не сохранить выдуманные поля. Если возникает конфликт между "показать больше данных" и "не дать мусору попасть в CRM", выбирать второе.

Запрещено в X-B:

- не менять `call_type` на `sales_call` или новый тип;
- не сохранять продукты, следующий шаг, возражения, бюджет и приоритет после non_conversation-обнуления;
- не создавать `sales_call_manual_review` как новый тип звонка;
- не отключать `_apply_non_conversation_hard_validation()`.

Итог: менеджер или аудитор увидит, что модель спорила с правилом, но CRM не получит выдуманные продажные поля.

## X-C. Будущая работа

X-C не реализуется в первом пакете X-v3. Scope сохраняется, но статус - эксперимент после X-A, X-D и X-B.

Причина: текущая обрезка длинного текста живет на `analyze.py:690-702`. Замена head+tail на умную середину влияет на то, какие факты модель увидит. Это нельзя включать без специальной выборки длинных тестовых транскриптов.

Если X-C будет запущен отдельно, делать только за новым полем в `src/mango_mvp/config.py`:

```python
analyze_transcript_chunking_mode: Literal["head_tail", "head_smart_middle_tail"] = "head_tail"
```

Но сейчас `Settings` - dataclass, поэтому нужно добавить поле в `Settings` на `config.py:50-115`, в `get_settings()` на `config.py:152-242` и во все тестовые конструкторы, где `Settings(...)` заполняется вручную, например `tests/test_dialogue_format.py:12-107`.

Важно для будущего X-C: в dataclass нельзя просто вставить новое поле с class default в середину списка обязательных полей и забыть про конструкторы. Текущий `Settings` создается явно в `get_settings()` и во многих тестах. Поэтому безопасный путь - добавить обязательное поле без class default, а default задавать в `get_settings()` через env и во всех `make_settings()` helpers. Это не нужно делать в X-v3 first package, но должно быть записано для будущей реализации.

Будущая acceptance для X-C должна быть только на искусственных fixture-файлах в `tests/fixtures/`, например `tests/fixtures/long_transcripts_for_chunking_ab.jsonl`. Нельзя измерять качество на `stable_runtime`, потому что это read-only зона и там параллельно идут другие разработки.

## Acceptance criteria

1. `_analysis_prompt_context()` для `profile="full"` возвращает `payload["user_prompt"]` с блоком `Deterministic hints JSON`.
2. `SYSTEM_PROMPT_FULL` явно разрешает deterministic hints только при подтверждении транскриптом.
3. `ANALYZE_PROMPT_VERSION_FULL == "v7"`, compact version остается `"v6"`, если compact prompt не менялся.
4. `_compose_history_summary()` добавляет коммерческий контекст, школу и hot/warm приоритет в обеих ветках: с LLM-черновиком и без него.
5. Пустые, дефолтные и шумовые значения не попадают в историю: нет "Бюджет: не указан", нет "Приоритет лида: cold".
6. Два отдельных `Клиент: Да.` подряд сохраняются; внутристрочное `"Да, да, да"` продолжает сжиматься.
7. При `RESOLVE_LLM_PROVIDER=off` `_resolve_with_llm()` возвращает `None`, а `llm_used` не увеличивается.
8. Compliance-преамбула плюс живой учебный диалог не приводит к `should_force_non_conversation=True`.
9. Чистый IVR, автоответчик, голосовая почта и виртуальный секретарь продолжают блокироваться как раньше.
10. При споре LLM с non_conversation-правилом появляется soft warning, но `call_type` остается `non_conversation`, а продажные поля остаются очищенными.

## Тесты

Новые или измененные тесты:

- `tests/test_analyze_xa_safe_pack.py::test_full_profile_user_prompt_includes_hints_section`
- `tests/test_analyze_xa_safe_pack.py::test_system_prompt_full_v7_mentions_hints`
- `tests/test_analyze_xa_safe_pack.py::test_compose_history_summary_adds_commercial_school_priority_with_draft`
- `tests/test_analyze_xa_safe_pack.py::test_compose_history_summary_adds_commercial_school_priority_without_draft`
- `tests/test_analyze_xa_safe_pack.py::test_compose_history_summary_skips_empty_budget_and_cold_priority`
- `tests/test_analyze_xa_safe_pack.py::test_consecutive_client_yes_lines_are_preserved`
- `tests/test_analyze.py::test_prompt_compaction_reduces_filler_without_losing_sales_content` - обновить старый assert на `Клиент: Да`.
- `tests/test_resolve.py::test_resolve_llm_provider_off_returns_none_immediately`
- `tests/test_resolve.py::test_resolve_llm_off_does_not_increment_llm_used`
- `tests/test_non_conversation_quality.py::test_compliance_preamble_with_live_sales_dialogue_is_not_forced_non_conversation`
- `tests/test_non_conversation_quality.py::test_pure_ivr_still_forces_non_conversation`
- `tests/test_analyze.py::test_non_conversation_llm_sales_signal_adds_soft_warning_without_preserving_fields`

Дополнительные проверки внутри тестов:

- для X-B проверять `result["quality_flags"]["call_type"] == "non_conversation"`;
- проверять, что `structured_fields["interests"]["products"] == []`;
- проверять, что `structured_fields["next_step"]["action"] is None`;
- проверять, что `quality_flags["non_conversation_soft_warning_llm_sales_signal"] is True`;
- не проверять `result["call_type"]`, потому что такого top-level ключа нет в текущем контракте.

Безопасный прогон:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q \
  tests/test_analyze.py \
  tests/test_analyze_xa_safe_pack.py \
  tests/test_non_conversation_quality.py \
  tests/test_resolve.py \
  tests/test_dialogue_format.py
```

Не запускать ASR/R+A, не запускать batch/start/run-ui скрипты, не читать `stable_runtime` как источник acceptance.

## Граничные условия

- Не менять `analysis_schema_version = "v2"`.
- Не менять контракт ключей ответа analyze: `history_summary`, `structured_fields`, `crm_blocks`, legacy-поля.
- Не менять зону ТЗ-Y: sanitizer, bot safety, tenant text normalizer.
- Не менять зону ТЗ-Z: transcribe merge suspicious drops, tenant_config pinning, hygiene smoke.
- Не добавлять новые типы звонков за пределами `CALL_TYPE_TAGS` на `analyze.py:153-159`.
- Не писать в AMO/Tallanto/CRM.
- Не использовать `stable_runtime` как тестовый набор.

## Использование субагентов

Можно использовать до 6 субагентов, но только с разделением зон:

1. X-A по `analyze.py` prompt/history/filler.
2. X-D по `resolve.py` и `config.py`.
3. X-B по `quality/non_conversation.py`.
4. Отдельный проверяющий по тестам и ручным конструкторам `Settings`.
5. Отдельный проверяющий по сохранению обратной совместимости analyze JSON.
6. Финальный reviewer только на чтение.

Субагентам нельзя менять одни и те же файлы одновременно. Если есть несколько workers, write-set должен быть разделен.

## Deliverables

Измененные файлы:

- `src/mango_mvp/services/analyze.py`
- `src/mango_mvp/services/resolve.py`
- `src/mango_mvp/config.py`
- `.env.example`
- `tests/test_analyze.py`
- новые тесты `tests/test_analyze_xa_safe_pack.py` и при необходимости `tests/test_non_conversation_xb_soft_warning.py`
- `docs/RESOLVE_LLM_DISABLED_2026-05-15.md`

Audit pack:

- `audits/_inbox/tz_x_v3_analyze_refactor_<timestamp>/implementation_notes.md`
- `audits/_inbox/tz_x_v3_analyze_refactor_<timestamp>/test_output.txt`
- `audits/_inbox/tz_x_v3_analyze_refactor_<timestamp>/risk_review.md`

## Backward compatibility

Должны остаться совместимыми:

- JSON-схема анализа `v2`;
- downstream чтение `structured_fields`, `crm_blocks`, legacy-полей;
- старые сценарии `non_conversation` для автоответчиков и IVR;
- режим compact prompt;
- поведение resolve при включенных `ollama`, `openai`, `codex_cli`;
- ручные конструкторы `Settings(...)` в тестах.
