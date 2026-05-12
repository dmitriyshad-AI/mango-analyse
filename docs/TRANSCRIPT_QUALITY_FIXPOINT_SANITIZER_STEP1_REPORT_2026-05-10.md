# Fixpoint Sanitizer: Step 1 Report

Дата: 2026-05-10

## Что сделано

Реализован первый архитектурный шаг после Claude/GPT-аудитов: sanitizer теперь работает до стабильного результата, а не одним проходом.

Изменения в коде:

- `sanitize_answer(...)` запускает `_sanitize_answer_pass(...)` до fixed point, максимум `MAX_SANITIZER_PASSES = 5`.
- Если за 5 проходов текст продолжает меняться, строка получает `status = fixpoint_not_reached`, `fixpoint_reached = False` и не может попасть в bot export.
- `SanitizedText` теперь хранит `pass_count` и `fixpoint_reached`.
- Knowledge Base пишет `bot_sanitizer_pass_count` и `bot_sanitizer_fixpoint_reached` в `enriched_reviews.csv` и summary.
- Stage 15 export gate считает `fixpoint_not_reached` отдельным safety-risk и блокирует такие строки.
- Добавлены regression-тесты на strong idempotence и на блокировку при недостижении fixed point.

## Проверки

Полный тестовый прогон:

```text
695 passed, 82 warnings
```

Stage 15 на пересобранном слое:

```text
passed: true
bot_export_allowlist_rows: 473
blocked_bot_export_rows: 0
stage14_residual_risk_rows: 0
stage14_over_sanitization_rows: 250
```

Risk-counts в финальном bot export:

```text
brand: 0
money_or_terms: 0
personal_data: 0
spoken_money_or_terms: 0
messenger_handle: 0
unsafe_placeholder: 0
brand_variant: 0
likely_single_name: 0
fixpoint_not_reached: 0
missing: 0
```

Risk-counts в исходных bot-safe колонках KB/ROP также все нулевые, включая `fixpoint_not_reached`.

## Pass-Count Distribution

На этапе генерации `enriched_reviews.csv`:

| Pass count | Rows |
|---:|---:|
| 1 | 2695 |
| 2 | 30 |
| 3 | 1 |

На финальном `bot_export_allowlist.csv` при повторной проверке:

| Pass count | Rows |
|---:|---:|
| 1 | 473 |

Strong idempotence на финальном allowlist:

```text
allowlist_idempotence_failures: 0
```

Интерпретация: fixed point сейчас стабилен. Массового 4-5 pass поведения нет, значит конфликтующих sanitizer-правил не видно.

## Проверка строк из Claude Re-Audit

Взяты все `rewrite_before_bot` строки из `claude_reaudit_row_decisions.csv`: 12 строк, покрывающие P1/P2 findings из re-audit.

Результат после пересборки:

```text
claude_reaudit_rewrite_rows_checked: 12
claude_reaudit_rows_with_after_risk: 0
```

Важно: в Claude findings было больше находок, чем строк, потому что часть строк содержала несколько проблем одновременно, например имя + кабинет или имя + улица.

## Новые артефакты

- `stable_runtime/fixpoint_sanitizer_step1_20260510/summary.json`
- `stable_runtime/fixpoint_sanitizer_step1_20260510/pass_count_distribution.csv`
- `stable_runtime/fixpoint_sanitizer_step1_20260510/claude_reaudit_before_after.csv`
- `stable_runtime/fixpoint_sanitizer_step1_20260510/allowlist_idempotence_failures.csv`

Пересобранные рабочие артефакты:

- `stable_runtime/sales_insight_knowledge_base_after_quality_backfill_20260510_v10_fixpoint/`
- `stable_runtime/rop_validation_pack_after_quality_backfill_20260510_v10_fixpoint/`
- `stable_runtime/transcript_quality_baseline_after_quality_backfill_20260510_v10_fixpoint/`
- `stable_runtime/transcript_quality_stage14_comparison_20260510_v7_fixpoint/`
- `stable_runtime/transcript_quality_stage15_export_gate_20260510_v10_fixpoint/`

## Ограничения

- Это закрывает structural bug класса “sanitizer сам создаёт новый контекст, который надо проверить повторно”.
- Это не заменяет frozen adversarial corpus и независимый detector. Они нужны следующими шагами, чтобы уйти от literal-whack-a-mole.
- `stage14_over_sanitization_rows = 250` по-прежнему блокирует автономного бота. Это не P0/P1 утечка, а очередь проверки полезности ответов после сильной санитизации.
- Средняя latency sanitizer может вырасти, но для batch Stage 14/15 это не критично. Перед online bot runtime нужно отдельно измерить latency.

## Вывод

Шаг 1 выполнен. Текущий export-layer стабилен, Stage 15 green, известных P0/P1/P2 утечек из последнего Claude re-audit в проверяемых строках не осталось.

Следующий шаг по плану: зафиксировать `THREAT_MODEL.md` и начать строить frozen adversarial corpus из трех слоев: synthetic, hand-curated audit findings, random real rows.
