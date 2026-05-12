# Transcript Quality Staged Backfill Report

Дата: 2026-05-09.

## Цель

Безопасно применить GPT-confirmed hard-gate `non_conversation` исправления после priority GPT-review, не затрагивая спорные звонки и не повторяя ASR.

## Новый staged-runner

Добавлен runner, который умеет применять `auto_apply_ready.csv` не к одной SQLite DB, а к нескольким DB сразу:

- `src/mango_mvp/quality/hard_gate_staged_backfill.py`
- `scripts/apply_hard_gate_staged_backfill.py`
- `tests/test_hard_gate_staged_backfill.py`

Он делает следующее:

- выбирает только `queue=auto_apply_ready`;
- режет строки по колонке `db`;
- для каждой SQLite DB создает отдельный `candidates.csv`;
- запускает существующую безопасную backfill-логику;
- в `apply` режиме создает backup каждой затронутой DB;
- сохраняет единый `summary.json`, `db_summary.csv` и per-DB отчеты.

## Stage 8: малая партия 100 строк

Dry-run:

`stable_runtime/non_conversation_hard_gate_staged_apply_20260509_stage100_dry_run/`

Итог:

- selected rows: 100
- selected DBs: 7
- planned updates: 100
- blocked rows: 0
- missing rows: 0
- errors: 0

Apply:

`stable_runtime/non_conversation_hard_gate_staged_apply_20260509_stage100_apply/`

Итог:

- applied updates: 100
- blocked rows: 0
- missing rows: 0
- errors: 0

Post-apply validation:

- validated rows: 100
- ok: 100
- problems: 0
- backup found for all 7 touched DBs

## Stage 9: весь первый GPT-reviewed subset, 480 строк

После успешной малой партии применен весь `priority01` safe subset.

Dry-run:

`stable_runtime/non_conversation_hard_gate_staged_apply_20260509_priority01_all480_dry_run/`

Итог:

- selected rows: 480
- selected DBs: 19
- planned updates: 380
- already applied: 100
- blocked rows: 0
- missing rows: 0
- errors: 0

Apply:

`stable_runtime/non_conversation_hard_gate_staged_apply_20260509_priority01_all480_apply/`

Итог:

- applied updates: 380
- already applied: 100
- blocked rows: 0
- missing rows: 0
- errors: 0

Post-apply validation:

- validated rows: 480
- ok: 480
- problems: 0
- verify dry-run: `already_applied=480`

## Extra: priority02 GPT-review и apply

Чтобы не останавливаться после первого чанка, выполнен GPT-review второго priority chunk.

GPT-review:

`stable_runtime/non_conversation_hard_gate_gpt_review_20260509_priority02/`

Итог:

- input tasks: 500
- reviews written: 500
- errors: 0
- `safe_apply`: 500
- model: `gpt-5.5`
- reasoning effort: `medium`

Combined decisions:

`stable_runtime/non_conversation_hard_gate_gpt_review_20260509_priority01_02_combined/`

Итог:

- total decisions: 1000
- `safe_apply`: 980
- `manual_review`: 11
- `keep_current`: 9
- duplicate task ids: 0

Apply-plan после двух чанков:

`stable_runtime/non_conversation_hard_gate_gpt_apply_plan_20260509_after_priority02_review/`

Очереди:

| Queue | Count |
|---|---:|
| `auto_apply_ready` | 980 |
| `gpt_review_required` | 4761 |
| `manual_review` | 11 |
| `keep_current` | 9 |

Apply по двум чанкам:

`stable_runtime/non_conversation_hard_gate_staged_apply_20260509_priority01_02_apply/`

Итог:

- selected rows: 980
- selected DBs: 21
- applied updates: 500
- already applied: 480
- blocked rows: 0
- missing rows: 0
- errors: 0

Post-apply validation:

- validated rows: 980
- ok: 980
- problems: 0
- verify dry-run: `already_applied=980`


## Extra: priority03 GPT-review и apply

GPT-review третьего priority chunk:

`stable_runtime/non_conversation_hard_gate_gpt_review_20260509_priority03/`

Итог:

- input tasks: 500
- reviews written: 500
- errors: 0
- `safe_apply`: 497
- `manual_review`: 2
- `keep_current`: 1
- model: `gpt-5.5`
- reasoning effort: `medium`

Combined decisions 01-03:

`stable_runtime/non_conversation_hard_gate_gpt_review_20260509_priority01_03_combined/`

Итог:

- total decisions: 1500
- `safe_apply`: 1477
- `manual_review`: 13
- `keep_current`: 10
- duplicate task ids: 0

Apply-plan после трех chunks:

`stable_runtime/non_conversation_hard_gate_gpt_apply_plan_20260509_after_priority03_review/`

Очереди:

| Queue | Count |
|---|---:|
| `auto_apply_ready` | 1477 |
| `gpt_review_required` | 4261 |
| `manual_review` | 13 |
| `keep_current` | 10 |

Apply по трем chunks:

`stable_runtime/non_conversation_hard_gate_staged_apply_20260509_priority01_03_apply/`

Итог:

- selected rows: 1477
- selected DBs: 21
- applied updates: 497
- already applied: 980
- blocked rows: 0
- missing rows: 0
- errors: 0

Post-apply validation:

- validated rows: 1477
- ok: 1477
- problems: 0
- verify dry-run: `already_applied=1477`


## Final: priority04-12 accelerated GPT-review и full auto-apply

После проверки стабильности `workers=2` была включена ускоренная схема: параллельные независимые chunks по 2 workers каждый.

GPT-review chunks 04-12:

| Chunk | Tasks | safe_apply | manual_review | keep_current | errors |
|---|---:|---:|---:|---:|---:|
| 04 | 500 | 493 | 5 | 2 | 0 |
| 05 | 500 | 498 | 1 | 1 | 0 |
| 06 | 500 | 490 | 5 | 5 | 0 |
| 07 | 500 | 499 | 0 | 1 | 0 |
| 08 | 500 | 500 | 0 | 0 | 0 |
| 09 | 500 | 491 | 9 | 0 | 0 |
| 10 | 500 | 470 | 23 | 7 | 0 |
| 11 | 500 | 493 | 4 | 3 | 0 |
| 12 | 261 | 261 | 0 | 0 | 0 |

Final combined decisions 01-12:

`stable_runtime/non_conversation_hard_gate_gpt_review_20260509_priority01_12_combined/`

Итог:

- total reviewed: 5761
- `safe_apply`: 5672
- `manual_review`: 60
- `keep_current`: 29
- duplicate task ids: 0
- GPT-review errors: 0

Final apply-plan:

`stable_runtime/non_conversation_hard_gate_gpt_apply_plan_20260509_after_priority12_review/`

Очереди:

| Queue | Count |
|---|---:|
| `auto_apply_ready` | 5672 |
| `manual_review` | 60 |
| `keep_current` | 29 |
| `gpt_review_required` | 0 |

Final apply:

`stable_runtime/non_conversation_hard_gate_staged_apply_20260509_priority01_12_apply/`

Итог:

- selected rows: 5672
- selected DBs: 25
- applied updates in final run: 1224
- already applied before final run: 4448
- blocked rows: 0
- missing rows: 0
- errors: 0

Final post-apply validation:

- validated rows: 5672
- ok: 5672
- problems: 0
- verify dry-run: `already_applied=5672`

## Что именно изменилось в SQLite

Для подтвержденных `safe_apply` строк обновлен `analysis_json`:

- `quality_flags.call_type = non_conversation`
- `history_summary` заменен на короткое объяснение, что живого диалога не было
- `follow_up_score = 0`
- `tags = [non_conversation]`
- `quality_flags.transcript_quality_backfill.version = safe_non_contentful_v1`
- `quality_flags.transcript_quality_backfill.source_gpt_decision = safe_apply`
- `sync_status = pending`
- `analysis_status = done`

Спорные строки не трогались:

- `manual_review`: 11
- `keep_current`: 9
- еще не reviewed: 4761

## Проверки качества

Фактическая проверка SQLite после apply:

- JSON парсится по всем проверенным строкам.
- Backfill metadata присутствует.
- `manual_review` и `keep_current` не применялись.
- Повторный dry-run не планирует повторную запись.

Тесты проекта после изменений:

```bash
PYTHONPATH=src python3 -m pytest
```

Результат: `665 passed`, `1 warning` (`urllib3`/LibreSSL, не относится к изменениям).

## Следующий шаг

Продолжать цикл по priority chunks:

1. GPT-review `priority04`.
2. Пересборка apply-plan с decisions 01-04.
3. Dry-run.
4. Apply только новых `auto_apply_ready`.
5. Post-apply validation.

После обработки всех chunks можно переходить к full downstream rebuild: readiness/contact-layer, knowledge base, ROP pack, bot seeds.
