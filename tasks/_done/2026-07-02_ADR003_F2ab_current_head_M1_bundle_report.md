# ADR-003 F2ab: current-HEAD M1 bundle for fresh frame/proof shadow measurement

Дата: 2026-07-02
Ветка: `codex/adr003-semanticframe-migration`
HEAD: `ae4843a6`

## Зачем нужен свежий прогон

Предыдущий валидный M1-прогон был на `36ea110`. Он подтвердил безопасность shadow (`route_text_diff=0`, `too_confident=0`, P0 не понижен), но показал `would_demote=0`.

После `36ea110` в текущем HEAD есть влияющие изменения:

- калибровка prompt границы `справка о существовании курса/формата != живое наличие мест/запись`;
- `existence_proof_shadow`;
- `proof_reconciliation_shadow`;
- отчёты F2y/F2z/F2aa, которые доказали `NO-GO` для route-only/action-only на старом сырье.

Значит старый `36ea110` остаётся достаточным для вывода `не включать active сейчас`, но недостаточен как финальный замер качества текущего SemanticFrame.

## Bundle

Папка:

`/Users/dmitrijfabarisov/Yandex.Disk.localized/OpenClaw/Actual Mango Tests/adr003_f2ab_clean_ae4843a6_20260702`

Файлы:

- `mango_clean_adr003_f2ab_ae4843a6.tar.gz`
- `BUNDLE_MANIFEST.json`
- `PROMT_M1_RUN.md`
- `SHA256SUMS.txt`

Контрольные суммы:

```text
c2979a3726cb41bc53dbeaececef908b597459822b3f3baf38bd5f7fd38c5ee4  mango_clean_adr003_f2ab_ae4843a6.tar.gz
9da0b15a2a7b12c42350b8952ca1d6cbbb5e62ebbe83273945b68df9d71cb9e6  BUNDLE_MANIFEST.json
9964f2d25f15e7955a55c0ed9e053bf4e00e0168d22e3d0e7ab8478a357670a3  PROMT_M1_RUN.md
```

Ключевые входы внутри архива:

- `product_data/telegram_dynamic_test_sets/adr003_semantic_frame_m1_scenarios_20260701.jsonl`
- `product_data/telegram_dynamic_test_sets/adr003_frame_gold_labels_20260701.jsonl`
- `product_data/knowledge_base/kb_release_20260612_v6_7_staging_r4_1/kb_release_v3_snapshot.json`

Snapshot SHA256:

```text
f2f211a56cafc213326d4841752e38a081172de867bd598b08724a510b065ba5
```

## Что должен сделать M1

Запустить `PROMT_M1_RUN.md`.

Прогон:

- OFF: текущий direct-path с `pilot_gold_v1`, `TELEGRAM_P0_MODEL_LED=1`, `TELEGRAM_PROSE_MODEL_LED=1`, frame/proof/self-answer shadows OFF.
- ON: enrich из OFF, frame/proof/reconciliation/self-answer shadows ON, active self-answer OFF.

Отчёты:

- `report/adr003_semantic_frame_eval_report.json`
- `gold_calibration/adr003_frame_gold_calibration_report.json`
- `overhandoff/adr003_overhandoff_levers_report.json`
- `frame_calibration_queue/adr003_frame_calibration_queue_report.json`
- `current_handoff_fact_gap/adr003_current_handoff_fact_gap_report.json`
- `partial_opportunities/adr003_partial_answer_opportunities_report.json`
- `action_counterfactual/adr003_action_gate_counterfactual_proof_report.json`
- `sha_manifest.json`

## Acceptance для регрейда Claude #1

Ожидаемые safety-гейты:

- `route_text_diff_count = 0`
- `p0_lowered_count = 0`
- `manager_only_lowered_count = 0`
- `money_lowered_count = 0`
- `operational_lowered_count = 0`
- `freshness_unknown_self_candidates = 0`
- `partial_freshness_self_candidates = 0`
- `too_confident = 0`

Диагностический вопрос:

- стало ли меньше `too_cautious` и `check_availability`-путаницы после prompt-калибровки;
- появились ли реальные `would_demote_to_self` кандидаты;
- если кандидаты есть, блокируются ли они fresh/exact fact proof или текстовой политикой.

## Safety

Это только M1/offline shadow-прогон. Живой бот, Telegram, AMO, Tallanto, CRM, профиль и live-флаги не трогать.

Live pid `60227` при сборке не трогался.
