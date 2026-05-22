# Semantic Review

Вердикт: `PASS_WITH_WARNINGS_FOR_CANDIDATE_PACKAGE`.

## Formal pass

- Пакет создан как read-only candidate package.
- Код, текущая KB, Telegram, почта, AMO, CRM, Tallanto и `stable_runtime` не менялись.
- Сырые письма, raw-вложения, raw-транскрипты, персональные значения и внешние ID не включены.
- Машинные CSV/JSONL содержат обезличенные синтетические формулировки и source refs.

## Semantic pass

`semantic_pass_for_candidate_package=true`, но только для внутреннего анализа и очереди решений.

`pilot_ready=false`.

`production_ready=false`.

`approved_for_kb_import=false`.

## Главные предупреждения

1. Исторические каналы не являются источником истины для фактов.
2. Хорошие ответы менеджеров можно использовать только как стиль и структуру.
3. В текущей KB обнаружены новые рискованные факты; они вынесены в `KB_COVERAGE_GAPS.md` и `rejected_or_unsafe_candidates.csv`, но не исправлялись.
4. Email-черновики остаются заблокированными до thread layer, recipient guard и no-send контура.
5. OCR/вложения можно использовать только как evidence-candidate и только после отдельного контроля.
6. До расширения пилота найденные KB-риски должны быть выведены из client-safe слоя или исправлены.

## Semantic gates to add later

- `historical_candidate_not_fact_gate`
- `p0_route_gate`
- `zero_collect_gate`
- `primary_source_gate`
- `freshness_gate`
- `brand_isolation_gate`
- `attachment_truth_gate`
- `identity_gate`
- `pii_masking_gate`
- `manager_phrase_style_only_gate`
- `autonomy_default_closed_gate`
- `regression_conversion_gate`
- `not_a_bot_disclosure_gate`
- `service_data_leak_gate`
- `kb_bad_fact_quarantine_gate`
- `paid_proxy_style_balance_gate`
- `brand_relation_canonical_gate`

## Итог

Пакет полезен как очередь для РОПа, будущих тестов и будущего ТЗ. Его нельзя напрямую импортировать в KB или подключать к клиентским ответам.
