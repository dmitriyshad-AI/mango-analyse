# TZ-118 Block D: clean D primary для ролей моно-звонков

Дата: 2026-06-16

## Источник

Источник D primary: `codex/tz118-d-primary-clean`, commit `b00b3bd`.

Этот источник выбран вместо веток `codex/tz118-group4-primary-d` и `codex/tz116-offline-understanding`, потому что clean-ветка содержит только офлайн-изменения:

- `src/mango_mvp/services/transcribe.py`;
- `tests/test_tz118_transcribe_d_primary.py`.

Грязные сестринские ветки не вливались.

## Решение

D primary означает:

- default остаётся `off`;
- `codex_selective` используется только для слабых правиловых случаев в офлайн-назначении ролей моно-звонка;
- если правило уверено, остаётся `rule_high_conf`;
- Codex CLI запускается через временный нейтральный `CODEX_HOME`;
- `OPENAI_API_KEY`, `OPENAI_ORG_ID`, `OPENAI_PROJECT` удаляются из env перед вызовом Codex;
- live AMO/Tallanto/CRM, ASR, Resolve+Analyze и `stable_runtime` не трогались.

## Segment guard

Отклонённый `segment_guard` не входит в clean-источник `b00b3bd`.

Причина, почему метрика `94.62%` сохраняется без перепрогона:

- принятая метрика относилась к raw `codex_selective`;
- `segment_guard=repair` был отклонён отдельно как ухудшающий слой;
- в clean D нет кода, который применяет `segment_guard`;
- тест `tests/test_tz118_transcribe_d_primary.py` проверяет, что `segment_guard_applied` не появляется в meta.

## Управляемость low-info фильтра

Добавлено конфигурационное поле:

```text
MONO_ROLE_LOW_INFO_FILTER_MODE
```

Default: `mark`.

Это фиксирует уже существующее фактическое поведение `getattr(..., "mark")` как явную настройку.

## Защита токена Codex

Проверки и правки:

- временная папка `codex_home_role_assignment_neutral*` создаётся как уникальный каталог вне репозитория через `tempfile.mkdtemp(...)`;
- права временной папки выставляются в `700`;
- папка `codex_home_role_assignment_neutral/` добавлена в `.gitignore` как дополнительная защита для старого/ручного пути;
- `CODEX_HOME_COPY_ALLOWLIST` копирует только `auth.json`, `rules`, `skills`, `models_cache.json`;
- runtime `config.toml` перезаписывается нейтральным конфигом с `service_tier = "fast"`;
- runtime `AGENTS.md` перезаписывается нейтральной инструкцией;
- после вызова Codex временная папка удаляется через `finally`;
- тест покрывает очистку временного Codex home;
- M1/бандлы не получают этот runtime home: штатные бандлы берут tracked-файлы, а runtime home создаётся вне репозитория и удаляется после вызова.

## C/E и ключи

Подтверждение по Группе 4:

- C primary использует офлайн-предсказания/предрасчитанные Codex labels через `scripts/run_tz121_question_catalog_c_hybrid_shadow.py`;
- active `question_catalog/theme_assigner_llm.py` с `OPENAI_API_KEY` не является путём принятого C primary в Группе 4;
- E primary не использует OpenAI API key;
- политика без OpenAI API key для принятой Группы 4 сохраняется.

## Итоговая матрица

| Блок | Итог |
|---|---|
| B — исход сделки | `primary` |
| E — бренд | `primary` |
| C — категории вопросов | `primary` |
| D — роли моно-звонков | `primary(clean)` |
| A — качество сделки | `shadow` |

## Проверки

До финального merge в `main` требуется регрейд Claude/Дмитрия. Проверки на ветке `codex/tz125-finalize-group4` выполнены.

Точечные D/config проверки:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_tz118_transcribe_d_primary.py

5 passed, 1 warning
```

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q \
  tests/test_smoke.py::SmokePipelineTest::test_get_settings_parses_float_env_values \
  tests/test_dialogue_format.py::DialogueFormatTest::test_rule_based_mono_role_assignment

2 passed, 1 warning
```

Регрессия `test_tz121_*`:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_tz121_*.py
```

Результат: `9 passed`.

Полный pytest:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q

3311 passed, 5 skipped, 1 warning in 54.21s
```

`git diff --check`: чисто.

## Что не делалось

- грязная ветка `codex/tz118-group4-primary-d` не вливалась;
- ветка `codex/tz116-offline-understanding` не вливалась;
- `segment_guard` не возвращался;
- live-записи и live-чтения внешних систем не запускались;
- ASR и Resolve+Analyze не запускались.
