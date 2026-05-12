# Transcript Quality Priority-01 GPT Review Report

Дата: 2026-05-09.

## Цель

Проверить первый приоритетный чанк `priority_chunks_500` перед staged backfill hard-gate `non_conversation`.

Это промежуточный этап между backup/rollback manifest и первой записью в SQLite: deterministic слой нашел кандидатов, но production-policy требует GPT-only подтверждения `safe_apply`.

## Что сделано

1. Добавлен отдельный GPT-review runner для hard-gate задач:
   - `src/mango_mvp/quality/hard_gate_gpt_review.py`
   - `scripts/run_hard_gate_gpt_review.py`
   - `tests/test_hard_gate_gpt_review.py`
2. Runner вызывает `codex exec` в read-only sandbox с JSON schema и сохраняет результаты инкрементально.
3. Исправлена подготовка временного `CODEX_HOME`: stale-копия auth/config теперь перезаписывается, если исходный `~/.codex` свежее или отличается по размеру.
4. Проведены smoke-прогоны:
   - 10 задач, `batch-size=5`, `workers=1`: 10/10, ошибок 0.
   - 20 задач, `batch-size=10`, `workers=1`: 20/20, ошибок 0.
   - 20 задач, `batch-size=10`, `workers=2`: 20/20, ошибок 0.
5. Проведен полный GPT-review первого приоритетного чанка на 500 задачах.

## Основной результат

Папка результата:

`stable_runtime/non_conversation_hard_gate_gpt_review_20260509_priority01/`

Файлы:

- `reviews.jsonl` — 500 GPT решений.
- `reviews.csv` — та же информация в табличном виде.
- `errors.csv` — пустой, ошибок нет.
- `summary.json` — машинная сводка.
- `HARD_GATE_GPT_REVIEW_REPORT.md` — отчет runner-а.

Итоги GPT-review:

| Decision | Count |
|---|---:|
| `safe_apply` | 480 |
| `manual_review` | 11 |
| `keep_current` | 9 |

Покрытие по риску:

| Risk | Count |
|---|---:|
| `critical` | 448 |
| `high` | 52 |

Технически:

- input tasks: 500
- reviews written: 500
- errors: 0
- provider calls: 50
- fallback single calls: 0
- model: `gpt-5.5`
- reasoning effort: `medium`
- batch size: 10
- workers: 2

## Обновленный apply-plan

Папка:

`stable_runtime/non_conversation_hard_gate_gpt_apply_plan_20260509_after_priority01_review/`

Итоговые очереди после загрузки 500 GPT-решений:

| Queue | Count |
|---|---:|
| `auto_apply_ready` | 480 |
| `gpt_review_required` | 5261 |
| `manual_review` | 11 |
| `keep_current` | 9 |

Важно: `auto_apply_ready=480` еще не означает, что изменения уже применены. Это только разрешенная очередь для staged apply.

## Rollback manifest для готового subset

Папка:

`stable_runtime/non_conversation_hard_gate_backup_manifest_20260509_after_priority01_review/`

Итог:

- selected rows: 480
- affected SQLite DB: 19
- rollback rows: 480
- missing rows: 0
- schema warnings: 0

Этот manifest покрывает только `auto_apply_ready` subset после первого GPT-review. Перед фактической записью можно сделать реальные копии этих 19 DB через `backup_copy_plan.sh`.

## Проверки

Целевые тесты:

```bash
PYTHONPATH=src python3 -m pytest tests/test_hard_gate_gpt_review.py tests/test_hard_gate_gpt_apply_plan.py tests/test_llm_review.py
```

Результат: `18 passed`.

## Что не делалось

- SQLite базы не изменялись.
- Backfill не применялся.
- Downstream слои не пересобирались.
- Остальные 5261 кандидатов еще не прошли GPT-review.

## Следующий безопасный шаг

Сделать staged apply на малой партии 50-100 строк из `auto_apply_ready`, но только после выполнения backup copy plan для затрагиваемых DB.

Рекомендуемый порядок:

1. Создать реальные backup-копии 19 DB из manifest subset.
2. Применить 50-100 `auto_apply_ready` строк.
3. Проверить SQL counts до/после.
4. Проверить 20 примеров из измененных строк.
5. Если все корректно, переходить к staged apply на 480 строк или продолжать GPT-review следующих priority chunks.
