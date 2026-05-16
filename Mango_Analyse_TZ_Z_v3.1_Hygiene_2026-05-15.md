# ТЗ-Z-v3.1: финальное ТЗ на hygiene, suspicious drops и tenant config pin

Дата: 2026-05-15
Адресат: Codex как исполнитель реализации
Основа: `Mango_Analyse_TZ_Z_v2_Hygiene_2026-05-15.md` и `Mango_Analyse_TZ_Z_v3_Delta_2026-05-15.md`

## Контекст и проблема

Z-трек закрывает инфраструктурные и гигиенические риски, которые не должны менять бизнес-логику обработки клиентов.

Первая проблема - при merge двух ASR-вариантов правило `_merge_texts()` может отбросить подозрительные куски, но это никак не отражается в итоговом `merge_meta`. Сейчас `_merge_texts()` находится в `src/mango_mvp/services/transcribe.py:1215-1265`. Она токенизирует оба текста на `transcribe.py:1223-1224`, нормализует токены на `transcribe.py:1230-1231`, запускает `difflib.SequenceMatcher(a=primary_norm, b=secondary_norm, autojunk=False)` на `transcribe.py:1232`, а потом при `delete` и `insert` отбрасывает chunk, если `_is_suspicious_chunk(chunk)` вернул `True` на `transcribe.py:1243-1251`.

Сейчас `_merge_texts()` возвращает только строку. `_merge_variant_pair()` на `transcribe.py:1882-2016` вызывает `_merge_texts()` в четырех местах: основной rule path на `transcribe.py:1935`, fallback после `ollama` на `transcribe.py:1957`, fallback после `codex_cli` на `transcribe.py:1979`, fallback после `openai` на `transcribe.py:2002`. `notes` в meta-словарях является строкой: `transcribe.py:1897`, `1906`, `1918`, `1931`, `1946`, `1968`, `1990`, `2013`. Его нельзя превращать в dict.

Вторая проблема - tenant config уже загружается и попадает в summary, но нет явной pin-проверки sha256. `src/mango_mvp/productization/tenant_config.py:20-36` умеет загрузить JSON и посчитать sha256. `tenant_config_summary()` на `tenant_config.py:58-73` отдает path, sha256, tenant_id, schema_version. Но файла `src/mango_mvp/productization/tenant_config_pinning.py` сейчас нет, и `DEFAULT_TENANT_CONFIG_PATH` тоже нет. Единственный текущий default path находится локально в `scripts/build_post_backfill_amo_ready_export.py:59-67` как `DEFAULT_TENANT_CONFIG`.

Третья проблема - разные скрипты имеют разные уровни риска. `scripts/run_crm_writeback_quality_gate.py` сейчас принимает `--tenant-config` на строке 51 и пишет summary на строке 212, но не имеет явного режима "живая запись / проверка". `scripts/build_post_backfill_amo_ready_export.py` принимает `--tenant-config` на строке 210 и пишет summary на строке 1208, но `tests/test_post_backfill_amo_ready_export.py` сейчас красный на `test_contact_summary_does_not_embed_full_latest_summary` (`tests/test_post_backfill_amo_ready_export.py:225-250`). Поэтому интеграцию pin-check во второй скрипт нельзя делать до стабилизации этого теста.

Z-3, Z-4, Z-5 в этом ТЗ не реализуются. Они остаются future-work с явными зависимостями: Z-3 ждет Y, Z-4 требует нового сканирования промптов после X, Z-5 ждет стабилизации catalog/deal-aware и аккуратного расширения smoke-теста.

## Z-1. Конкретные правки suspicious_drops

Файл: `src/mango_mvp/services/transcribe.py:1215-1265`, `src/mango_mvp/services/transcribe.py:1882-2016`.

### Z-1.1. Не менять тип notes

Во всех словарях, которые возвращает `_merge_variant_pair()`, поле `notes` остается строкой. Новое поле добавляется как соседнее:

```python
"suspicious_drops": {
    "count": 0,
    "total_chars": 0,
    "samples": [],
}
```

Для веток, где merge не запускался или текст пустой (`secondary_empty`, `primary_empty`, `skip_high_similarity`, `primary`), допустимо возвращать zero summary. Для веток, где `_merge_texts()` реально применялся, нужно считать suspicious drops.

### Z-1.2. Счетчик должен повторять реальный merge

Не использовать несуществующую функцию `_normalize_tokens_for_merge`. Ее нет в текущем коде.

Добавить helper внутри `TranscribeService`, например:

```python
def _count_suspicious_drops_in_merge(self, primary_text: str, secondary_text: str) -> dict[str, Any]:
    primary_tokens = self._tokenize(primary_text)
    secondary_tokens = self._tokenize(secondary_text)
    primary_norm = [self._normalize_token(t) for t in primary_tokens]
    secondary_norm = [self._normalize_token(t) for t in secondary_tokens]
    matcher = difflib.SequenceMatcher(a=primary_norm, b=secondary_norm, autojunk=False)
    ...
```

Это должно точно повторять `_merge_texts()` на `transcribe.py:1223-1232`. Для samples использовать исходные токены, а не нормализованные, чтобы отчет был читаемым. Для opcodes:

- при `delete`: взять `primary_tokens[a0:a1]`, если `_is_suspicious_chunk(chunk)` вернул `True`, увеличить счетчик;
- при `insert`: взять `secondary_tokens[b0:b1]`, если `_is_suspicious_chunk(chunk)` вернул `True`, увеличить счетчик;
- при `replace`: ничего не считать как drop, потому что `_merge_texts()` выбирает лучший chunk через `_pick_better_chunk()` на `transcribe.py:1238-1241`, а не просто выкидывает.

Схема результата:

```python
{
    "count": int,
    "total_chars": int,
    "samples": list[str],
}
```

Ограничить samples до 5 элементов, каждый sample - не длиннее 200 символов. `total_chars` считать по исходным sample-строкам до обрезки, чтобы показатель не занижался.

Production call sites `_merge_variant_pair()` находятся на `transcribe.py:2319`, `transcribe.py:2324` и `transcribe.py:2483`. Итоговый dict результата попадает в `merge_meta`: manager на `transcribe.py:2410`, client на `transcribe.py:2416`, full на `transcribe.py:2546`. Поэтому новое поле `suspicious_drops` должно быть JSON-serializable и не должно содержать сложных объектов, Path, regex match или bytes.

Для LLM-success веток (`ollama`, `codex_cli`, `openai`) `_merge_texts()` не вызывается. В этих ветках нужно добавить zero summary `{"count": 0, "total_chars": 0, "samples": []}`, если провайдер сам не вернул `suspicious_drops`. Если провайдер вернул поле с неправильной схемой, нормализовать его к безопасной схеме, но не падать.

### Z-1.3. Не менять merged_text

Z-1 - это только диагностика. Результат `"text"` из `_merge_variant_pair()` должен остаться тем же, что был до правки для тех же входов и настроек.

Чтобы избежать расхождения между счетчиком и merge, можно сделать дополнительный приватный helper, который возвращает opcodes и токены, но менять публичный контракт `_merge_texts(primary_text, secondary_text) -> str` нельзя.

## Z-2. Конкретные правки tenant config pin

Файлы: `src/mango_mvp/productization/tenant_config.py:20-73`, новый `src/mango_mvp/productization/tenant_config_pinning.py`, `scripts/run_crm_writeback_quality_gate.py:44-64`, `scripts/run_crm_writeback_quality_gate.py:209-220`.

### Z-2.1. Создать новый модуль pinning

Создать файл `src/mango_mvp/productization/tenant_config_pinning.py`. В нем не использовать `DEFAULT_TENANT_CONFIG_PATH`, потому что такого символа нет.

Минимальный контракт:

```python
EXPECTED_TENANT_CONFIG_SHA256 = "9de1e6363171ea619cdd52055ce16f0b2b71c499a6d00fd88b2f55e70711c288"
EXPECTED_TENANT_CONFIG_SCHEMA_VERSION = "tenant_config_v1"
EXPECTED_TENANT_ID = "foton"

def check_tenant_config_pin(
    load_result: TenantConfigLoadResult | None,
    *,
    expected_sha256: str = EXPECTED_TENANT_CONFIG_SHA256,
    expected_tenant_id: str = EXPECTED_TENANT_ID,
    expected_schema_version: str = EXPECTED_TENANT_CONFIG_SCHEMA_VERSION,
) -> dict[str, Any]:
    ...
```

Возврат должен быть простым словарем, пригодным для JSON summary:

```python
{
    "passed": bool,
    "reason": str,
    "expected_sha256": str,
    "actual_sha256": str,
    "path": str,
    "tenant_id": str,
    "schema_version": str,
}
```

Если `load_result is None`, возвращать `passed=False`, `reason="tenant_config_not_loaded"`.

Если sha256 не совпал, возвращать `passed=False`, `reason="tenant_config_sha256_mismatch"`.

Если tenant_id или schema_version не совпали, возвращать `passed=False` с отдельной причиной.

### Z-2.2. Сделать рабочую команду print-current

В `tenant_config_pinning.py` добавить `if __name__ == "__main__":` с параметрами:

```bash
python -m mango_mvp.productization.tenant_config_pinning --print-current --path <path/to/tenant_config_v1.json>
```

`--path` обязателен для `--print-current`. Не импортировать default path из `tenant_config.py`. Если нужен путь по умолчанию для конкретного скрипта, он остается внутри этого скрипта.

Команда должна печатать:

- path;
- sha256;
- tenant_id;
- schema_version;
- строку, какой constant обновить при осознанном изменении конфига.

Тест не должен зависеть от реального archive path. Использовать `tmp_path` и временный tenant_config.

### Z-2.3. Ввести явный режим pin-check в quality gate

В `scripts/run_crm_writeback_quality_gate.py:44-55` добавить аргумент:

```python
parser.add_argument("--tenant-config-pin-mode", choices=("off", "warn", "strict"), default="warn")
```

Логика:

- `off`: pin-check не выполняется, но `tenant_config` summary остается как сейчас.
- `warn`: pin-check выполняется; результат пишется в summary; несовпадение не ломает запуск.
- `strict`: pin-check выполняется; если `passed=False`, скрипт должен завершиться кодом 1 или выбросить понятную ошибку до записи "passed": true.

Так как `run_crm_writeback_quality_gate.py` сам по себе не пишет в AMO, default должен быть `warn`, чтобы не ломать локальные проверки. Для live-writeback runbook команда должна использовать `--tenant-config-pin-mode strict`.

В summary рядом с `tenant_config` на `run_crm_writeback_quality_gate.py:212` добавить:

```python
"tenant_config_pin": pin_summary,
"tenant_config_pin_mode": args.tenant_config_pin_mode,
```

Если `strict` не пройден, `summary["passed"]` должен быть `False`, даже если остальные проверки зеленые.

Для tests не использовать реальный Foton archive path. Тесты должны создавать временный config через `tmp_path`, загружать его через `load_tenant_config()`, а expected hash брать из результата или намеренно подставлять неправильный hash. Это делает тесты воспроизводимыми на любой машине.

### Z-2.4. build_post_backfill_amo_ready_export пока не трогать, если тест красный

Перед интеграцией в `scripts/build_post_backfill_amo_ready_export.py` нужно прогнать:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_post_backfill_amo_ready_export.py
```

Если тест красный на текущей проблеме истории контакта, Z-2 в этот скрипт не внедрять. В audit pack написать: "integration deferred because post_backfill test is red". Это не провал Z-v3.1, а защита от смешивания двух независимых работ.

Когда тест станет зеленым, добавить тот же аргумент `--tenant-config-pin-mode` в `scripts/build_post_backfill_amo_ready_export.py:195-211`. Default - `warn`, потому что скрипт строит export/preview, а не сам пишет в AMO. Strict использовать только в явной команде перед живой записью.

## Future-work: Z-3, Z-4, Z-5

Z-3 отложен до завершения Y, потому что процентный sanitizer сейчас меняется в Y-A.

Z-4 отложен до завершения X и нового сканирования prompt-мест. На 2026-05-15 простое сканирование `SYSTEM_PROMPT|PROMPT_TEMPLATE|PROMPT_VERSION` показывает больше 12 совпадений, включая `src/mango_mvp/question_catalog/theme_assigner_llm.py`. Перед Z-4 нужно заново сформировать список и не полагаться на старые 11 промптов.

Z-5 отложен до стабилизации catalog/deal-aware в git. `tests/test_smoke.py` нельзя просто расширить тяжелыми проверками; нужен легкий smoke без ASR/R+A и без записи в CRM.

## Acceptance criteria

1. `_merge_variant_pair()` возвращает `notes` как строку во всех ветках.
2. `_merge_variant_pair()` возвращает соседнее поле `suspicious_drops` со схемой `count`, `total_chars`, `samples`.
3. `_count_suspicious_drops_in_merge()` использует `_tokenize()`, `_normalize_token()` и `SequenceMatcher(..., autojunk=False)`, как `_merge_texts()`.
4. Для 20+ синтетических merge cases итоговый `"text"` до и после диагностической правки не меняется.
5. На cases с known ASR artifacts `suspicious_drops["count"] > 0`.
6. На чистых cases `suspicious_drops["count"] == 0`.
7. `src/mango_mvp/productization/tenant_config_pinning.py` создан и не импортирует несуществующий `DEFAULT_TENANT_CONFIG_PATH`.
8. `python -m mango_mvp.productization.tenant_config_pinning --print-current --path <tmp_config>` работает в тесте.
9. `run_crm_writeback_quality_gate.py` пишет `tenant_config_pin` и `tenant_config_pin_mode` в summary.
10. В режиме `strict` несовпадение sha256 блокирует quality gate.
11. В режиме `warn` несовпадение sha256 не ломает dry-run/preview, но отражается в summary.
12. `build_post_backfill_amo_ready_export.py` не меняется, если `tests/test_post_backfill_amo_ready_export.py` остается красным.
13. В `merge_meta` новое поле сериализуется в JSON без дополнительных преобразований.
14. LLM-success ветки получают zero `suspicious_drops`, если реального rule-drop не было.

## Тесты

Новые тесты Z-1:

- `tests/test_transcribe_suspicious_drops.py::test_merge_variant_pair_reports_suspicious_drops_without_changing_text`
- `tests/test_transcribe_suspicious_drops.py::test_suspicious_drops_counter_uses_normalized_tokens_and_autojunk_false`
- `tests/test_transcribe_suspicious_drops.py::test_clean_merge_has_zero_suspicious_drops`
- `tests/test_transcribe_suspicious_drops.py::test_notes_remains_string_when_suspicious_drops_present`
- `tests/test_transcribe_suspicious_drops.py::test_llm_success_paths_add_zero_suspicious_drops`
- `tests/test_transcribe_suspicious_drops.py::test_merge_meta_with_suspicious_drops_is_json_serializable`

Фикстуры:

- `tests/fixtures/transcribe_merge_corpus_z1/*.json`

В фикстурах минимум 20 пар. Они должны быть синтетическими, без production audio/transcripts из `stable_runtime`.

Новые тесты Z-2:

- `tests/test_tenant_config_pinning.py::test_tenant_config_pin_passes_for_expected_hash`
- `tests/test_tenant_config_pinning.py::test_tenant_config_pin_fails_on_hash_mismatch`
- `tests/test_tenant_config_pinning.py::test_tenant_config_pin_fails_when_config_not_loaded`
- `tests/test_tenant_config_pinning.py::test_print_current_cli_outputs_hash_for_explicit_path`
- `tests/test_crm_writeback_quality_gate.py::test_quality_gate_warns_on_tenant_config_pin_mismatch`
- `tests/test_crm_writeback_quality_gate.py::test_quality_gate_strict_blocks_tenant_config_pin_mismatch`

Безопасный прогон:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q \
  tests/test_transcribe_suspicious_drops.py \
  tests/test_tenant_config.py \
  tests/test_tenant_config_pinning.py \
  tests/test_crm_writeback_quality_gate.py \
  tests/test_llm_review_merge.py
```

Отдельная проверка перед вторым скриптом:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_post_backfill_amo_ready_export.py
```

Если она красная, integration в `build_post_backfill_amo_ready_export.py` откладывается.

## Граничные условия

- Не менять публичный контракт `_merge_texts(primary_text, secondary_text) -> str`.
- Не превращать `notes` в dict.
- Не запускать ASR/R+A.
- Не читать production audio/transcripts из `stable_runtime` для acceptance.
- Не писать в AMO/Tallanto/CRM.
- Не менять `scripts/write_deal_aware_amo_fields.py` в рамках Z-v3.1.
- Не менять `tests/test_smoke.py` в рамках Z-v3.1.
- Не внедрять pin-check в `build_post_backfill_amo_ready_export.py`, пока его тест красный.

## Последовательность реализации

Шаг 1. Создать audit pack и зафиксировать текущий статус. В audit pack записать:

- `git status --short` по файлам Z;
- результат `tests/test_llm_review_merge.py`;
- результат `tests/test_tenant_config.py`;
- результат `tests/test_crm_writeback_quality_gate.py`;
- текущий результат `tests/test_post_backfill_amo_ready_export.py`.

Если `test_post_backfill_amo_ready_export.py` красный, это не блокирует Z-1 и pin-check в quality gate, но блокирует любые изменения `scripts/build_post_backfill_amo_ready_export.py`.

Шаг 2. Сделать Z-1 через тесты. Сначала добавить synthetic fixtures и tests, которые проверяют текущий текст merge. Потом добавить `_count_suspicious_drops_in_merge()` и нормализацию схемы `suspicious_drops`. После этого проверить, что `text` не меняется. Если текст меняется, это ошибка: Z-1 не должен улучшать merge, он только добавляет диагностику.

Шаг 3. Покрыть все return paths `_merge_variant_pair()`. Для веток без merge - zero summary. Для rule/fallback веток - реальный счетчик. Для LLM-success веток - zero summary или нормализованное поле провайдера. Все `notes` остаются строками.

Шаг 4. Сделать новый модуль `tenant_config_pinning.py` с тестами на `tmp_path`. Нельзя начинать с реального Foton path, иначе тест будет зависеть от локальной машины и архива. Реальный hash Foton можно хранить как constant для production pin, но unit tests должны уметь передавать expected hash явно.

Шаг 5. Подключить pin-check к `run_crm_writeback_quality_gate.py` в режиме `warn` по умолчанию. Затем добавить `strict` tests. Если strict падает, summary должен ясно объяснять причину: config не загружен, hash mismatch, tenant mismatch или schema mismatch.

Шаг 6. Еще раз отдельно прогнать `tests/test_post_backfill_amo_ready_export.py`. Если он всё еще красный, в audit pack записать deferred decision и не трогать второй скрипт. Если стал зеленым из-за параллельной разработки, можно подключить pin-check к `build_post_backfill_amo_ready_export.py` тем же способом, но только отдельным маленьким diff.

## Риски реализации

Главный риск Z-1 - случайно изменить сам merge. Это недопустимо, потому что suspicious drops сейчас нужен как диагностический слой, а не как новая логика распознавания. Любое изменение `merged_text` должно считаться регрессией, кроме случаев, где тест явно доказывает старую ошибку и Дмитрий отдельно согласовал поведенческую правку.

Второй риск - сделать счетчик, который не совпадает с `_merge_texts()`. Если использовать raw tokens или default `SequenceMatcher`, отчет будет красивым, но неверным. Поэтому helper должен повторять `_tokenize`, `_normalize_token` и `autojunk=False` буквально.

Третий риск - превратить pin-check в ломатель локальных проверок. Поэтому default `warn`, а `strict` включается явно для живого контура. Это не снижает безопасность: live/runbook обязан использовать `strict`, а локальные тесты остаются удобными.

Четвертый риск - смешать pin-check с текущей красной разработкой post-backfill истории. Это запрещено. Красный тест по истории контакта чинится отдельно, не внутри Z-v3.1.

## Использование субагентов

Можно использовать до 6 субагентов:

1. Z-1 worker: только `transcribe.py` и `tests/test_transcribe_suspicious_drops.py`.
2. Fixture worker: только `tests/fixtures/transcribe_merge_corpus_z1/`.
3. Z-2 worker: только `tenant_config_pinning.py` и `tests/test_tenant_config_pinning.py`.
4. Quality gate worker: только `scripts/run_crm_writeback_quality_gate.py` и его tests.
5. Verification worker: прогон безопасных тестов и проверка красного/зеленого post_backfill.
6. Reviewer: финальный read-only audit.

Write-set должен быть разделен. Никто не должен править stable_runtime.

## Deliverables

Измененные файлы:

- `src/mango_mvp/services/transcribe.py`
- `src/mango_mvp/productization/tenant_config_pinning.py`
- `scripts/run_crm_writeback_quality_gate.py`
- `tests/test_transcribe_suspicious_drops.py`
- `tests/test_tenant_config_pinning.py`
- `tests/test_crm_writeback_quality_gate.py`
- `tests/fixtures/transcribe_merge_corpus_z1/`

Условно изменяемый файл, только если его тест зеленый:

- `scripts/build_post_backfill_amo_ready_export.py`

Audit pack:

- `audits/_inbox/tz_z_v31_hygiene_<timestamp>/merge_suspicious_drops_cases.md`
- `audits/_inbox/tz_z_v31_hygiene_<timestamp>/tenant_config_pin_summary.md`
- `audits/_inbox/tz_z_v31_hygiene_<timestamp>/test_output.txt`
- `audits/_inbox/tz_z_v31_hygiene_<timestamp>/post_backfill_status.txt`

## Backward compatibility

Должны остаться совместимыми:

- `_merge_texts()` возвращает строку;
- `_merge_variant_pair()` сохраняет старые ключи `text`, `selection`, `confidence`, `provider`, `notes`, `similarity`;
- `notes` остается строкой;
- старые consumers, которые не знают о `suspicious_drops`, продолжают работать;
- `load_tenant_config()` и `tenant_config_summary()` сохраняют текущий контракт;
- dry-run/preview quality gate не ломается из-за pin mismatch при default `warn`;
- live/writeback runbook может включить `strict` явно.
