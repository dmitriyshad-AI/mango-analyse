# Tenant Text Normalizer Quality System

## Цель

Не чинить `НПК МФТИ`, `МПК МФТИ`, `летние ночные школы` и похожие ошибки точечно после каждого ручного замечания, а держать постоянный quality-loop для закрытых классов ASR/LLM-артефактов.

## Что можно гарантировать

Детерминистический normalizer может надежно закрывать только классы, которые формально описаны:

- tenant brand aliases;
- типовые ASR-искажения продуктов;
- счетчики и технические артефакты в продуктовых списках;
- синонимы продуктов;
- повторяющиеся возражения с одинаковым смыслом.

Он не может гарантировать исправление любого неизвестного будущего ASR-искажения. Для этого нужен audit-loop и выборочный LLM/human review.

## Контур качества

1. **Taxonomy**
   Каждый новый тип ошибки сначала оформляется как класс: `tenant_brand_alias`, `summer_night_school_asr_artifact`, `product_count_artifact`, `objection_dedupe`, а не как единичная строка.

2. **Frozen corpus**
   Для каждого класса есть набор adversarial-примеров:

   `tests/fixtures/tenant_text_normalizer_frozen_corpus.jsonl`

   Туда добавляются:

   - реальные найденные примеры;
   - синтетические варианты;
   - ASR-искажения с похожим звучанием;
   - negative guards, чтобы normalizer не портил корректный текст.

3. **Unit tests**
   `tests/test_tenant_text_normalizer.py` прогоняет frozen corpus и проверяет expected output / forbidden substrings.

4. **Population gate**
   `scripts/run_tenant_text_normalizer_gate.py` сканирует CSV-артефакты и падает, если после нормализации остаются закрытые residual artifacts.

   Gate отдельно считает:

   - `pre_normalization_findings`: что было бы исправлено normalizer-ом в текущем артефакте;
   - `residual_findings`: что осталось сломанным даже после normalizer-а.

   Для production-артефактов нужен не только `residual_findings=0`, но и `pre_normalization_findings=0`, иначе CSV уже собран со старым текстом и должен быть пересобран.

5. **Downstream application**
   Normalizer должен применяться во всех manager-facing слоях:

   - ROP/manual review packs;
   - AMO-ready export;
   - AMO writeback payload;
   - future deal-aware contact/deal summaries;
   - future bot/KB manager-facing outputs.

6. **Audit loop**
   Периодически берем случайную выборку строк, где detector ничего не нашел, и отдаем Claude/GPT на поиск false negatives. Если найден новый класс, он добавляется в taxonomy + frozen corpus + detector.

## Exit criterion для класса

Класс можно считать закрытым только если:

- все frozen-corpus cases проходят;
- population gate на текущих рабочих CSV дает `residual_findings = 0`;
- на negative guards нет порчи корректных формулировок;
- один независимый аудит на случайной выборке не находит этот же класс снова.

## Текущий статус

Реализовано:

- `src/mango_mvp/quality/tenant_text_normalizer.py`
- `tests/fixtures/tenant_text_normalizer_frozen_corpus.jsonl`
- `tests/test_tenant_text_normalizer.py`
- `scripts/run_tenant_text_normalizer_gate.py`

Покрытые классы:

- `tenant_brand_alias`: `МПК/НПК/ОМПК/ВНПК/МНПК/УНИПК МФП/УНПК МФП` и близкие ASR-варианты `-> УНПК МФТИ`;
- `summer_night_school_asr_artifact`: `летние ночные школы -> летние очные школы`;
- `product_count_artifact`: убрать `(N касаний)` и `label: N`;
- `objection_dedupe`: схлопывать повторные смысловые возражения.

## Последняя проверка

После расширения класса `tenant_brand_alias` был пересобран свежий downstream:

- `stable_runtime/sales_master_export_20260513_human_history_v8_normalized/`
- `stable_runtime/student_card_manual_review_next50_20260513_v6_normalized/`

Population gate:

`stable_runtime/tenant_text_normalizer_gate_20260513_v2_after_rebuild/summary.json`

Результат на 16 027 строках и 475 158 ячейках:

- `passed=true`
- `pre_normalization_findings=0`
- `residual_findings=0`
- `class_counts={}`

Это означает: текущие свежие ROP/AMO/master CSV уже не содержат закрытых tenant-normalizer классов ошибок.
