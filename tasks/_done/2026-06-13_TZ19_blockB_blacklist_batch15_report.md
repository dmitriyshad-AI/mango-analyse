# TZ-19 Block B: blacklist batch 15

Дата: 2026-06-13

## Scope

Малый батч 15 blacklist-звонков прогнан локально на исправленной full v7 analyze-подсказке. Это единственное место ТЗ-19, где были разрешены LLM-вызовы.

Запрещённые действия не выполнялись:

- полный перепрогон 77 не запускался;
- в canonical DB ничего не записывалось;
- AMO/Tallanto/CRM write не выполнялся;
- ASR и Resolve+Analyze не запускались.

## Input

Blacklist source:

`/Users/dmitrijfabarisov/Yandex.Disk.localized/OpenClaw/Actual Mango Tests/analyze_rerun_20260611/blacklist_77.txt`

Исключены звонки микропробы ТЗ-16:

`12617`, `14115`, `14327`, `15112`, `16146`

Детерминированно выбраны первые 15 оставшихся id:

`16628`, `19055`, `19871`, `21767`, `22259`, `24772`, `26435`, `27025`, `27191`, `28320`, `28911`, `29433`, `34882`, `35059`, `37544`

Local ignored artifact root:

`product_data/customer_profiles/tz19_blacklist_batch15/`

Full result export for Claude review:

`product_data/customer_profiles/tz19_blacklist_batch15/results_batch15.jsonl.gz`

Prompt sha256 сверяется с бандлом A:

`12718ea6b8a5ee500910300c4c2de7c3695f78217c3b63a62d572de612b5eacf`

## Run

Runner:

`scripts/run_analyze_ab_test.py`

CLI mode:

`--cli $(which python3)`, `PYTHONPATH=src:.`

Arm:

`mini_v7:gpt-5.4-mini:full`

Metrics:

- total: `15`
- done: `15`
- failed: `0`
- pending: `0`
- analysis_model: `gpt-5.4-mini` for `15/15`
- analysis_prompt_version: `v7` for `15/15`
- elapsed_sec: `196.385`
- `llm_calls_total=15`

## Review Table

| call_id | duration_sec | call_type before | call_type after | target_product before | target_product after |
|---:|---:|---|---|---:|---:|
| 16628 | 529 | service_call | service_call | false | false |
| 19055 | 1800 | sales_call | service_call | true | true |
| 19871 | 1557 | sales_call | service_call | true | true |
| 21767 | 88 | service_call | service_call | false | false |
| 22259 | 83 | service_call | non_conversation | false | false |
| 24772 | 86 | technical_call | non_conversation | false | false |
| 26435 | 686 | sales_call | service_call | true | true |
| 27025 | 62 | service_call | non_conversation | false | false |
| 27191 | 730 | sales_call | service_call | true | true |
| 28320 | 61 | sales_call | non_conversation | true | false |
| 28911 | 1000 | sales_call | service_call | true | true |
| 29433 | 231 | sales_call | non_conversation | true | false |
| 34882 | 115 | service_call | service_call | false | false |
| 35059 | 136 | sales_call | non_conversation | false | false |
| 37544 | 95 | sales_call | non_conversation | true | false |

Aggregate:

- `service_call` after: `8`
- `non_conversation` after: `7`
- target_product_present after: `6`

## NEG

Three true autoresponder controls from TZ-16 were checked without LLM through deterministic non-conversation guardrails:

| call_id | old_call_type | recommended_call_type | force_non_conversation | label |
|---:|---|---|---:|---|
| 15717 | non_conversation | non_conversation | true | non_conversation_high_confidence |
| 16565 | non_conversation | non_conversation | true | non_conversation_high_confidence |
| 24790 | non_conversation | non_conversation | true | non_conversation_high_confidence |

## Semantic Status

`formal_pass`: batch completed, metadata is v7/mini, artifacts are ignored.

`semantic_pass`: `PASS_WITH_NOTES`. The batch confirms the anti-autoresponder fix is not a simple universal switch: long meaningful calls are recovered, while short or system-like cases can still be `non_conversation`. Full 77 rerun remains gated by Claude/Dmitry review of `results_batch15.jsonl.gz`.
