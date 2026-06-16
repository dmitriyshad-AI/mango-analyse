# TZ-118 Block D: D primary для ролей моно-звонков

Дата: 2026-06-16

## Решение

Принято решение Дмитрия после регрейда: `segment_guard=repair` не применять, потому что он дал `net -165`.

D primary теперь означает:

- где правиловое назначение ролей уверено, остаётся правило (`rule_high_conf`);
- где правило слабое, включается `codex_selective`;
- транспорт модели только Codex CLI, без `OPENAI_API_KEY`;
- `low_info=mark` допустим как диагностическая метка и не меняет роли;
- `segment_guard` в `primary` принудительно `off`, даже если в CLI случайно передали `--segment-guard-mode repair`.

Это только офлайн-аналитика. Живой путь бота, AMO, Tallanto, CRM, ASR и `stable_runtime` не трогались.

## Изменённые файлы

- `scripts/run_tz116_mono_role_gold50_measure.py`
- `scripts/run_tz116_mono_role_shadow_real.py`
- `tests/test_tz116_offline_modes.py`
- `tasks/_done/2026-06-16_TZ118_block_D_primary_enable_report.md`
- `tasks/_done/2026-06-16_TZ118_block_D_primary_semantic_review.md`

## Что изменено

- Снят старый стоппер `primary is blocked` в двух D-runner'ах после принятого регрейда.
- `primary` теперь реально вызывает тот же безопасный путь `codex_selective`, что и `shadow`.
- Счётчик `llm_calls_total` теперь считает вызовы Codex и для `primary`.
- В отчётах добавлены признаки:
  - `primary_scope=offline_low_confidence_codex_selective`;
  - `segment_guard_forced_off_in_primary=true`;
  - `segment_guard_mode=off`.
- Для real-runner добавлен `--low-info-filter-mode`, default `mark`; режим `off` по-прежнему не вызывает модель и не меняет поведение.

## Проверки

Точечные:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_tz116_offline_modes.py::test_real_mono_shadow_runner_default_off_reads_sqlite_without_assignment tests/test_tz116_offline_modes.py::test_real_mono_shadow_runner_shadow_uses_codex_selective_without_openai_api tests/test_tz116_offline_modes.py::test_real_mono_primary_uses_codex_selective_after_regrede_without_segment_guard tests/test_tz116_offline_modes.py::test_mono_role_gold50_measure_calls_codex_only_for_low_confidence tests/test_tz116_offline_modes.py::test_mono_role_gold50_primary_forces_segment_guard_off tests/test_tz116_offline_modes.py::test_mono_role_gold50_measure_reports_segment_guard_net_effect tests/test_dialogue_format.py::DialogueFormatTest::test_segment_guard_default_off_keeps_codex_roles_byte_for_byte
```

Результат: `7 passed, 1 warning`.

Расширенные:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_tz116_offline_modes.py tests/test_dialogue_format.py tests/test_smoke.py::SmokePipelineTest::test_get_settings_parses_float_env_values
```

Результат: `44 passed, 1 warning`.

Полные:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q
```

Результат: `3300 passed, 5 skipped, 1 warning`.

## NEG

- `primary` не использует OpenAI API и не требует ключей.
- `primary` не пишет в БД.
- `primary` не пишет в AMO/Tallanto/CRM.
- `primary` не запускает ASR.
- `primary` не включает `segment_guard`, даже если передали `--segment-guard-mode repair`.
- `off` остаётся режимом по умолчанию.

## Счётчики

- Новых live-вызовов модели в этом коммите: `0`.
- Опора на принятый регрейд: raw `codex_selective` на gold-наборе дал `94.62%` среднюю точность по репликам против около `55%` у слабого правила.
- Остаточный риск: около `6%` модельных сегментных путаниц остаются известным ограничением и не чинятся правилом в этом блоке.

## Следующий шаг

Можно переходить к следующему блоку TZ-118 по утверждённому порядку. Если потребуется улучшать оставшиеся ошибки D, делать это отдельным модельным улучшением, а не возвратом к `segment_guard=repair`.
