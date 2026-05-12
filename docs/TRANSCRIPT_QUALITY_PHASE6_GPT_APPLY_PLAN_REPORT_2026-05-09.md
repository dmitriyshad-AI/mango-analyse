# Transcript Quality Phase 6 GPT Apply Plan Report

Дата: 2026-05-09.

## Что сделано

Построен full-corpus GPT-only apply-plan по кандидатам phase 5.

Вход:

`stable_runtime/non_conversation_hard_gate_owner_dry_run_20260509_v5_gpt_policy_preview/would_update_candidates.csv`

Выход:

`stable_runtime/non_conversation_hard_gate_gpt_apply_plan_20260509_phase6/`

Базы данных не изменялись. Это read-only планирование.

## Итог

| Очередь | Количество |
|---|---:|
| gpt_review_required | 5 761 |
| auto_apply_ready | 0 |
| blocked_candidates | 0 |
| keep_current | 0 |
| manual_review | 0 |

Почему `auto_apply_ready=0`: GPT-decisions по полному корпусу ещё не загружены. По новой production-policy auto-apply возможен только после GPT `safe_apply` + deterministic high-confidence guardrail.

## Риск-приоритеты

| Risk | Count |
|---|---:|
| critical | 448 |
| high | 3 693 |
| medium | 869 |
| low | 751 |

`critical` в первую очередь включает старые `sales_call` и `existing_client_progress`, потому что у них максимальный риск ошибочно стереть содержательный клиентский разговор.

## Основные артефакты

- `full_apply_plan.csv` — полный план по всем 5 761 строкам.
- `gpt_review_required.csv` — очередь GPT-review.
- `auto_apply_ready.csv` — сейчас пустой, заполнится после пересборки с GPT decisions.
- `blocked_candidates.csv` — сейчас пустой, значит все deterministic candidates валидны для GPT-review.
- `gpt_review_tasks.jsonl` — полный GPT-review пакет с транскриптами.
- `gpt_decisions_template.jsonl` — шаблон ожидаемых ответов.
- `GPT_REVIEW_PROMPT_RU.md` — инструкция для GPT-review.
- `month_summary.csv` — разбивка по месяцам.
- `risk_summary.csv` — разбивка по риску.
- `gpt_review_shards/` — risk-specific shards и 12 priority chunks по 500 задач.

## Как использовать дальше

1. Сначала подготовить backup/rollback manifest для затрагиваемых БД.
2. Запустить GPT-review по `gpt_review_shards/priority_chunks_500/`, начиная с первого чанка.
3. Объединить GPT decisions в один JSONL.
4. Пересобрать apply-plan:

```bash
PYTHONPATH=src python3 scripts/build_hard_gate_gpt_apply_plan.py \
  --project-root . \
  --candidates-csv stable_runtime/non_conversation_hard_gate_owner_dry_run_20260509_v5_gpt_policy_preview/would_update_candidates.csv \
  --gpt-decisions-jsonl PATH_TO_GPT_DECISIONS.jsonl \
  --out-root stable_runtime/non_conversation_hard_gate_gpt_apply_plan_after_review_YYYYMMDD
```

5. Использовать только `auto_apply_ready.csv` как вход для staged backfill.
6. Не применять ничего в SQLite без backup/rollback manifest.
