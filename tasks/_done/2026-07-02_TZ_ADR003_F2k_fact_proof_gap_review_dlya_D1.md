Ветка: codex/adr003-semanticframe-migration
Зоны: audits/_inbox/, tasks/_done/
Тест-команда: report-only, без изменения runtime-кода
Семантический-аудит: да

# ADR-003 F2k: проверка остаточного рычага после prompt-калибровки

## Контекст

После F2j prompt-калибровка SemanticFrame исправила главный частный перекос:
`requested_action` на existence/format subset перестал массово превращаться в
`check_availability`.

Остаточный блокер остался в другом месте: `risk_class=missing_facts`,
`answerability=manager_only`, `must_handoff=true`.

## Что проверено

На F2j subset повторно прогнаны существующие report-only scorers:

- `scripts/report_adr003_existence_fact_verification.py`;
- `scripts/report_adr003_fact_gated_self_answer_readiness.py`;
- `scripts/report_adr003_exact_proof_injection_shadow.py`.

Входы:

- transcripts: `/tmp/adr003_f2j_posthoc_measure/dynamic_dialog_transcripts.jsonl`;
- gold: `/tmp/adr003_f2j_existence_subset_gold.jsonl`;
- KB snapshot: `product_data/knowledge_base/kb_release_20260612_v6_7_staging_r4_1/kb_release_v3_snapshot.json`.

Артефакты сохранены в:

`audits/_inbox/adr003_f2k_fact_proof_gap_review_20260702074500/local_fact_reports/`.

## Результат

На F2j subset:

- `requested_action` стал лучше: старое `check_availability` ушло с subset;
- `strict_f3_draft_candidates=0`;
- `current_handoff_rows=0` в existence/readiness scorer на этом subset;
- `manager_only_exact_proof_rows=0`;
- active readiness остаётся `no_go`.

Сырой просмотр транскриптов показывает, что оставшиеся спорные строки часто уже
идут как `bot_answer_self_for_pilot`, но frame всё равно телеметрически
пессимистичен. Там, где ответ реально неполный, у runtime нет `exact_fact_keys`,
а self-answer shadow блокирует по `no_exact_fact_keys` или `missing_facts`.

## Вердикт

F2k не должен делать новый prompt-bypass.

Если насильно заставить SemanticFrame писать `safe/answer_self` без проверенного
факта, мы улучшим метрику frame, но ослабим анти-выдумку. Это противоречит
инварианту ADR-003: модель понимает, детерминизм верифицирует.

Следующий реальный шаг автономности:

1. доставить в runtime телеметрию проверенного evidence/fact proof для
   existence/format вопросов;
2. отдельно отличать:
   - стабильное существование/формат продукта;
   - живые места/подходящую группу/запись;
3. только после этого снова мерить route-only active candidates.

Поведение бота, профиль, P0-floor, live-процессы и внешние системы не тронуты.
