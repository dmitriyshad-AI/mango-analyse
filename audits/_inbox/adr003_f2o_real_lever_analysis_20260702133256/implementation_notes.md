# ADR003 F2o Real Lever Analysis

## Изменение

Расширен отчёт `scripts/report_adr003_frame_calibration_queue.py`:

- добавлен блок `real_lever_analysis`;
- все true `too_cautious` строки разбиваются по текущему route, frame `requested_action`,
  risk/answerability и классу рычага;
- отдельно считаются `fact_assertion_required`, `factless_ack_status`,
  `danger_adjacent`, `already_self_or_no_route_leverage`,
  `clean_route_only_discussion`.

Runtime/direct-path не менялся.

## Почему

После регрейда F2/F2n стало ясно, что активный Ф3 нельзя готовить по price или
harmless ack/status. Нужно сначала честно показать, где реальный рычаг:
route-only или слой проверки существования факта.

## Сырьё

Код отчёта: `f1a00156`.

Вход: paired no-op F2n enrichment на M1/F2 наборе:

- transcripts:
  `audits/_inbox/adr003_f2n_enrich_existence_proof_noop_20260702130739/ON/dynamic_dialog_transcripts.jsonl`;
- gold:
  `product_data/telegram_dynamic_test_sets/adr003_frame_gold_labels_20260701.jsonl`;
- KB snapshot:
  `product_data/knowledge_base/kb_release_20260612_v6_7_staging_r4_1/kb_release_v3_snapshot.json`.

## Результат

- true frame too-cautious: 14;
- current handoff among too-cautious: 6;
- fact assertion required: 12;
- factless ack/status: 0;
- danger-adjacent: 4;
- stable existence misread as check_availability: 0;
- stable existence misread as enroll: 1;
- true live availability negative controls: 29;
- true enroll/booking negative controls: 9;
- clean route-only discussion rows: 0;
- strict active candidates now: 0;
- true too-confident: 0.

## Вывод

Ф2o подтверждает NO-GO для active route demotion. Реальный следующий рычаг -
не route-only, а калибровка SemanticFrame и проверенный путь фактов для
существования/формата курса без обещания живых мест.
