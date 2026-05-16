# ТЗ-Y-v3.1: финальное ТЗ на sanitizer и tenant text quality

Дата: 2026-05-15
Адресат: Codex как исполнитель реализации
Основа: `Mango_Analyse_TZ_Y_v2_Sanitizer_Quality_2026-05-14.md` и `Mango_Analyse_TZ_Y_v3_Delta_2026-05-15.md`

## Контекст и проблема

Y-трек закрывает две связанные, но независимые проблемы.

Первая проблема - sanitizer слишком агрессивно маскирует числа и проценты. В `src/mango_mvp/insights/sanitizers.py:23-38` `MONEY_AMOUNT_RE` ловит реальные цены, что правильно, но две альтернативы дают ложные срабатывания:

- standalone spaced-thousands на `sanitizers.py:26`: `5 000 человек`, `2 500 баллов`;
- `за NNNN` на `sanitizers.py:33`: `за 5000 человек`.

Эти фразы не являются ценой, но сейчас превращаются в `[CURRENT_PRICE]`. При этом нельзя переписать весь `MONEY_AMOUNT_RE`: в нем уже есть нужные альтернативы для `50к`, `100 т.р.`, `пятьдесят тысяч рублей`, `стоимость 50000`, `7900 за 4 занятия`, `год целиком за 147000`.

Вторая проблема - `PERCENT_RE` на `sanitizers.py:39-43` маскирует любой процент, включая неценовые фразы вроде `100% результат` и `100 процентов результат`. Это вредно для базы ответов бота: текст про результат обучения или посещаемость может быть испорчен placeholder-ом `[PAYMENT_OPTIONS]`. При этом скидочные проценты должны по-прежнему маскироваться: `10 процентов`, `15% скидка`, `скидка 10%`.

Третья проблема - tenant normalizer уже улучшался в рабочем дереве, но часть хвостовых ASR-вариантов бренда не нормализуется и не ловится детектором. В `src/mango_mvp/quality/tenant_text_normalizer.py:8-13` есть `BRAND_ALIASES_RE`, в `tenant_text_normalizer.py:39-47` есть только `normalize_manager_text()`. Функции `normalize_customer_text()` в текущем коде нет, поэтому ее нельзя упоминать как место правки. Структура `TenantTextArtifact` определена на `tenant_text_normalizer.py:32-36` и имеет поля `class_id`, `matched_text`, `reason`. Старый пример из Y-v3 с `artifact_type` и `sample` неверен и при буквальной реализации сломает код.

Y-C в этом ТЗ не выполняется. Причина: изменение default severity для `has_blocking_crm_text_quality_risk` не имеет смысла без ROP-аудита реальных P2-строк и без производственных caller-ов, которые этим default пользуются.

## Y-A. Конкретные правки sanitizer

Файл: `src/mango_mvp/insights/sanitizers.py:23-43`, применение на `sanitizers.py:240-250`.

### Y-A.1. Money: точечно защитить non-money counts

Не переписывать `MONEY_AMOUNT_RE` целиком. Добавить рядом с регулярками отдельный набор non-money units, например:

```python
NON_MONEY_UNIT_RE = (
    r"человек(?:а|у|ом|е)?|люд(?:и|ей|ям|ьми|ях)?|"
    r"ученик(?:а|ов|у|ам|ами|ах)?|дет(?:и|ей|ям|ьми|ях)?|"
    r"клиент(?:а|ов|у|ам|ами|ах)?|балл(?:а|ов|у|ам|ами|ах)?|"
    r"мест(?:о|а|ам|ами|ах)?|заявк(?:а|и|ок|е|ам|ами|ах)?|"
    r"сообщени(?:е|я|й|ю|ям|ями|ях)?|касани(?:е|я|й|ю|ям|ями|ях)?"
)
```

Список должен быть достаточно узким. Не добавлять туда `год`, `семестр`, `занятие`, `урок`, `курс`, `смена`, `месяц`, потому что эти слова уже используются в денежных конструкциях на `sanitizers.py:34-35`.

Изменить только две альтернативы:

1. `sanitizers.py:26`: после spaced-thousands добавить `(?!\s*(?:{NON_MONEY_UNIT_RE})\b)`.
2. `sanitizers.py:33`: после `\bза\s+\d{4,6}\b` добавить `(?!\s*(?:{NON_MONEY_UNIT_RE})\b)`, сохранив существующее исключение для года.

Остальные альтернативы на `sanitizers.py:24-25`, `sanitizers.py:27-36` не менять.

Ожидаемое поведение:

- `5 000 человек` не содержит `[CURRENT_PRICE]`;
- `за 5000 человек` не содержит `[CURRENT_PRICE]`;
- `2 500 баллов` не содержит `[CURRENT_PRICE]`;
- `50 000 рублей` содержит `[CURRENT_PRICE]`;
- `50к` содержит `[CURRENT_PRICE]`;
- `100 т.р.` содержит `[CURRENT_PRICE]`;
- `пятьдесят тысяч рублей` содержит `[CURRENT_PRICE]`;
- `стоимость 50000` содержит `[CURRENT_PRICE]`;
- `7900 за 4 занятия` продолжает содержать `[CURRENT_PRICE]`.

### Y-A.2. Percent: сохранить скидки, не портить результативность

Файл: `src/mango_mvp/insights/sanitizers.py:39-43`.

Добавить рядом отдельный узкий контекст для нескидочных процентов:

```python
NON_DISCOUNT_PERCENT_CONTEXT_RE = (
    r"результат\w*|успех\w*|гарант\w*|охват\w*|"
    r"посещаем\w*|сдач\w*|выполнен\w*|готовност\w*"
)
```

Применить его только к числовой альтернативе `\d{1,3}\s*(?:%|процент(?:а|ов)?)`. То есть:

- `100% результат` не маскируется;
- `100 процентов результат` не маскируется;
- `100% гарантия результата` не маскируется как платежный риск;
- `10 процентов` маскируется как раньше;
- `15% скидка` маскируется как раньше;
- `скидка 10%` маскируется как раньше;
- словесные проценты `десять процентов`, `пятнадцать процентов` маскируются как раньше.

Важно: `sanitize_answer()` на `sanitizers.py:241` сначала заменяет процент на `[PERCENT]`, а затем на `sanitizers.py:250` превращает `[PERCENT]` в `[PAYMENT_OPTIONS]`. Поэтому тесты должны проверять `[PAYMENT_OPTIONS]`, а не `[PERCENT]`.

Старый слабый assert вида `assert "[CURRENT_PRICE]" in result.text or "actual" not in result.text.lower()` запрещен. Нужны прямые проверки.

Дополнительно проверить публичный слой question catalog. `src/mango_mvp/question_catalog/safety.py` использует `sanitize_answer()` через `redact_public_text()` и `assert_public_text_safe()`. Если `100% результат` перестал считаться платежным риском в sanitizer, публичный слой тоже не должен блокировать такую фразу. Но `15% скидка`, цена, телефон и email по-прежнему должны редактироваться или блокироваться.

Не менять `src/mango_mvp/quality/bot_safety_detector.py` в рамках Y-v3.1. Там есть независимые `MONEY_RE/PERCENT_RE`, и его задача шире: ловить небезопасные ответы бота. Если после исправления sanitizer отдельный bot safety detector продолжит считать `100% результат` риском, это фиксируется как отдельная будущая задача, а не смешивается с Y-v3.1. В этом ТЗ меняется sanitizer и tenant text normalizer, а не весь слой bot safety.

## Y-B. Конкретные правки tenant normalizer

Файл: `src/mango_mvp/quality/tenant_text_normalizer.py:8-13`, `tenant_text_normalizer.py:32-36`, `tenant_text_normalizer.py:39-73`.

Перед началом реализации обязательно сохранить текущий diff:

```bash
git diff HEAD -- src/mango_mvp/quality/tenant_text_normalizer.py tests/test_tenant_text_normalizer.py \
  > audits/_inbox/tz_y_v31_sanitizer_quality_<timestamp>/pending_changes_before_yb.patch
```

Это важно, потому что `tenant_text_normalizer.py` уже изменен в рабочем дереве. Работать нужно поверх текущего состояния, не откатывая чужие или параллельные изменения.

### Y-B.1. Нормализовать хвостовые варианты УНПК МФТИ

Добавить отдельную точечную регулярку для известных хвостовых ошибок:

- `УНПК МФТИШ`
- `УНПК МФТИК`
- `УНПК МФТИЙ`
- `УНПК МФТИВ`
- `УНПК МФТИНГ`

Применять ее только в `normalize_manager_text()` после `BRAND_ALIASES_RE.sub("УНПК МФТИ", text)` на `tenant_text_normalizer.py:44`. Не добавлять `normalize_customer_text()`, потому что такой функции нет.

Правило должно нормализовать эти варианты в `УНПК МФТИ`, но не портить каноническую фразу `УНПК МФТИ подготовка`.

### Y-B.2. Усилить detector без неверных полей

Текущий `detect_residual_manager_text_artifacts()` на `tenant_text_normalizer.py:50-73` ищет только `BRAND_ALIASES_RE` и summer night school artifacts. Нужно добавить два слоя:

1. `DETECTOR_KNOWN_BRAND_VARIANTS` - явный список известных остаточных вариантов.
2. `DETECTOR_BRAND_GENERAL_PATTERNS` - узкие catch-all patterns для новых странных вариантов.

При создании findings использовать только реальные поля:

```python
TenantTextArtifact(
    class_id="known_brand_variant_residual",
    matched_text=match.group(0),
    reason="ASR/LLM brand artifact must be normalized to УНПК МФТИ",
)
```

Не использовать `artifact_type` и `sample`.

Для known variants нельзя делать простой `variant.lower() in text.lower()`: нужна проверка с границами слова, например через `re.finditer(rf"(?<!\w){re.escape(variant)}(?!\w)", text, re.I)`. Это защищает от ложного срабатывания внутри более длинного слова.

Детектор не должен ругаться на нормальную фразу `УНПК МФТИ подготовка`.

Если один и тот же кусок пойман explicit-list и general-pattern, нужно убрать дубли. Достаточно дедуплицировать по `(class_id, matched_text.casefold())` или по span+class_id.

Детектор должен использоваться как проверка остаточного мусора после нормализации. Поэтому для строки `normalize_manager_text("УНПК МФТИШ подготовка")` итог должен быть `УНПК МФТИ подготовка`, а `detect_residual_manager_text_artifacts()` на этом результате должен вернуть пустой список. Для строки, которую не нормализовали специально в тесте, например прямой вызов detector на `"УНПК МФТИШ подготовка"`, должен быть finding с `matched_text`, содержащим исходный вариант.

Важно не расширять regex так широко, чтобы он начал ловить обычные слова вокруг бренда. Главный false positive, который нельзя допустить: `УНПК МФТИ подготовка` и похожие канонические фразы.

## Acceptance criteria

1. `sanitize_answer("5 000 человек", mode="bot").text` не содержит `[CURRENT_PRICE]`.
2. `sanitize_answer("за 5000 человек", mode="bot").text` не содержит `[CURRENT_PRICE]`.
3. `sanitize_answer("2 500 баллов", mode="bot").text` не содержит `[CURRENT_PRICE]`.
4. Денежные регрессии продолжают работать: `50к`, `100 т.р.`, `пятьдесят тысяч рублей`, `стоимость 50000`, `50 000 рублей`, `7900 за 4 занятия`.
5. `100% результат` и `100 процентов результат` не превращаются в `[PAYMENT_OPTIONS]`.
6. `10 процентов`, `десять процентов`, `15% скидка`, `скидка 10%` продолжают маскироваться.
7. `normalize_manager_text()` превращает хвостовые варианты `УНПК МФТИШ/К/Й/В/НГ` в `УНПК МФТИ`.
8. `detect_residual_manager_text_artifacts()` ловит все known variants, если они остались после нормализации.
9. `detect_residual_manager_text_artifacts("УНПК МФТИ подготовка")` возвращает пустой список.
10. Все findings являются экземплярами `TenantTextArtifact` с заполненными `class_id`, `matched_text`, `reason`.
11. `redact_public_text("100% результат")` не вставляет `[PAYMENT_OPTIONS]`, а `assert_public_text_safe("100% результат")` не должен падать только из-за процента.
12. `redact_public_text("15% скидка")` продолжает редактировать платежный риск.
13. `has_money_or_terms_risk()` после исправления не считает `5 000 человек` деньгами, но продолжает считать `50 000 рублей`, `50к`, `10 процентов` и `15% скидка` риском.
14. Y-C не меняется.

## Тесты

Новые тесты лучше добавить в отдельный файл `tests/test_sanitizer_context_exclusions.py`, чтобы не раздувать `tests/test_knowledge_base.py`.

Тесты Y-A:

- `test_sanitize_answer_keeps_non_money_counts`
- `test_sanitize_answer_preserves_money_amount_regressions`
- `test_sanitize_answer_keeps_non_discount_percent_context`
- `test_sanitize_answer_preserves_discount_percent_forms`
- `test_question_catalog_safety_allows_result_percent_context`
- `test_question_catalog_safety_still_redacts_discount_percent`

Что проверяют:

```python
assert "[CURRENT_PRICE]" not in sanitize_answer("5 000 человек", mode="bot").text
assert "[CURRENT_PRICE]" in sanitize_answer("стоимость 50000", mode="bot").text
assert "[PAYMENT_OPTIONS]" not in sanitize_answer("100% результат", mode="bot").text
assert "[PAYMENT_OPTIONS]" not in sanitize_answer("100 процентов результат", mode="bot").text
assert "[PAYMENT_OPTIONS]" in sanitize_answer("10 процентов", mode="bot").text
assert "[PAYMENT_OPTIONS]" in sanitize_answer("15% скидка", mode="bot").text
```

Для `has_money_or_terms_risk()` добавить прямые проверки:

```python
assert has_money_or_terms_risk("5 000 человек") is False
assert has_money_or_terms_risk("за 5000 человек") is False
assert has_money_or_terms_risk("50 000 рублей") is True
assert has_money_or_terms_risk("15% скидка") is True
```

Тесты Y-B добавить в `tests/test_tenant_text_normalizer.py`:

- `test_normalizes_mfti_tail_variants`
- `test_detector_flags_known_brand_variants_with_real_artifact_fields`
- `test_detector_uses_word_boundaries_for_known_variants`
- `test_detector_does_not_flag_canonical_unpk_mfti`

В `tests/fixtures/tenant_text_normalizer_frozen_corpus.jsonl` можно добавить 5-10 новых строк, но не заменять существующие.

Безопасный прогон:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q \
  tests/test_sanitizer_context_exclusions.py \
  tests/test_tenant_text_normalizer.py \
  tests/test_knowledge_base.py \
  tests/test_question_catalog_safety.py \
  tests/test_bot_safety_detector.py \
  tests/test_bot_safety_frozen_corpus.py
```

Если `tests/test_bot_safety_detector.py` падает только из-за независимого `PERCENT_RE` в `src/mango_mvp/quality/bot_safety_detector.py`, не расширять Y-v3.1 автоматически. Нужно вынести это в отдельную заметку audit pack: "bot_safety_detector has independent percent policy". Менять его можно только после отдельного решения, потому что это уже не sanitizer, а gate автономной безопасности бота.

## Граничные условия

- Не переписывать весь `MONEY_AMOUNT_RE`.
- Не менять `DISCOUNT_RE`, `INSTALLMENT_RE`, `REFUND_RE`, `DEADLINE_RE`, если тесты не докажут прямую необходимость.
- Не менять `sanitize_answer()` pipeline на `sanitizers.py:182-257`, кроме новой логики regex.
- Не добавлять `normalize_customer_text()`.
- Не менять Y-C.
- Не использовать данные из `stable_runtime` как acceptance fixture.
- Не менять документы РОПа и question catalog в рамках этого ТЗ.
- Не менять `src/mango_mvp/quality/bot_safety_detector.py` без отдельного решения.

## Последовательность реализации

Реализация должна идти маленькими шагами, чтобы было понятно, какая правка сломала тест, если что-то пойдет не так.

Шаг 1. Создать audit pack и сохранить pending diff по tenant normalizer. Это делается до любых правок, потому что файл уже изменен в рабочем дереве. В audit pack также записать текущие результаты короткой проверки:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 - <<'PY'
from mango_mvp.insights.sanitizers import sanitize_answer
for text in ["5 000 человек", "за 5000 человек", "100% результат", "100 процентов результат"]:
    print(text, "=>", sanitize_answer(text, mode="bot").text)
PY
```

Шаг 2. Добавить failing tests для Y-A без изменения кода. Минимум четыре теста должны сначала показать текущую проблему: non-money counts маскируются, нескидочный процент маскируется. После этого менять regex. Если тесты сразу зеленые, значит текущая рабочая копия уже изменилась параллельной разработкой; в audit pack нужно записать "already fixed before implementation" и не дублировать правку.

Шаг 3. Исправить только `MONEY_AMOUNT_RE` и `PERCENT_RE`. После этого прогнать только sanitizer tests. Если падает `tests/test_knowledge_base.py`, сначала понять, это реальная регрессия или старый тест закреплял слишком широкое поведение. Нельзя "чинить" падение расширением scope на bot safety detector.

Шаг 4. Добавить failing tests для Y-B: хвостовые варианты `УНПК МФТИШ/К/Й/В/НГ` должны сначала показать проблему. Затем добавить нормализацию и detector. После этого прогнать `tests/test_tenant_text_normalizer.py`.

Шаг 5. Прогнать общий безопасный набор из раздела "Тесты". В audit pack записать команды и итог. Если какой-то тест красный из-за независимого слоя безопасности бота, не чинить это внутри Y-v3.1, а записать отдельный риск.

## Риски реализации

Главный риск Y-A - слишком широкий список non-money units. Если туда добавить слова, которые участвуют в цене, например `занятие`, можно сломать фразы вида `7900 за 4 занятия`. Поэтому список должен быть узким и подтверждаться регрессионными тестами.

Второй риск - слишком широкий процентный контекст. Если исключить все проценты перед словом `результат`, можно случайно пропустить скидочную фразу "скидка 100% результат акции" или похожий мусор. Поэтому исключение применяется только к ближайшему контексту после процента, а скидочные слова должны продолжать ловиться через `DISCOUNT_RE`.

Третий риск - detector начнет ловить канонический бренд. Это хуже, чем пропустить один редкий хвостовой вариант, потому что начнет засорять quality reports. Поэтому canonical false positive обязательно проверяется отдельным тестом.

Четвертый риск - смешать manager-facing normalizer и public bot safety. Y-v3.1 не должен решать все задачи автономного бота. Его цель уже: не портить хорошие ответы и не пропускать известный tenant-мусор.

## Чек-лист ревью перед сдачей

Перед финальным отчетом исполнитель должен вручную открыть diff и проверить четыре вещи:

1. В `MONEY_AMOUNT_RE` изменены только две опасные альтернативы, а не весь regex.
2. В `PERCENT_RE` словесная альтернатива `десять процентов` не удалена.
3. В `TenantTextArtifact(...)` нигде не появились поля `artifact_type` или `sample`.
4. В реализации не появилась функция `normalize_customer_text()`.

Если любой из этих пунктов нарушен, работу нельзя сдавать как Y-v3.1. Нужно откатить именно свою последнюю правку через новый patch, не трогая чужие изменения, и привести diff обратно к scope этого ТЗ.

## Использование субагентов

Можно использовать до 6 субагентов:

1. Проверка денежных regex-регрессий.
2. Проверка процентных regex-регрессий.
3. Проверка tenant normalizer и detector.
4. Проверка frozen corpus.
5. Проверка bot safety.
6. Финальный reviewer по diff и тестам.

Если подключаются workers, один должен владеть только `sanitizers.py` и sanitizer tests, второй - только `tenant_text_normalizer.py` и его tests. Нельзя двум workers одновременно менять один файл.

## Deliverables

Измененные файлы:

- `src/mango_mvp/insights/sanitizers.py`
- `src/mango_mvp/quality/tenant_text_normalizer.py`
- `tests/test_sanitizer_context_exclusions.py`
- `tests/test_tenant_text_normalizer.py`
- возможно `tests/fixtures/tenant_text_normalizer_frozen_corpus.jsonl`

Audit pack:

- `audits/_inbox/tz_y_v31_sanitizer_quality_<timestamp>/pending_changes_before_yb.patch`
- `audits/_inbox/tz_y_v31_sanitizer_quality_<timestamp>/test_output.txt`
- `audits/_inbox/tz_y_v31_sanitizer_quality_<timestamp>/regex_cases.md`

## Backward compatibility

Должны остаться совместимыми:

- `SanitizedText` dataclass и поля `text`, `flags`, `status`, `pass_count`, `fixpoint_reached`;
- режимы sanitizer `"manager"`, `"bot"`, `"customer"`;
- существующие placeholders `[CURRENT_PRICE]`, `[PAYMENT_OPTIONS]`, `[CURRENT_DEADLINE]`, `[CLIENT_NAME]` и остальные;
- `TenantTextArtifact(class_id, matched_text, reason)`;
- текущий API `normalize_manager_text()`, `format_product_list()`, `format_objection_list()`;
- все текущие тесты `tests/test_knowledge_base.py`.
