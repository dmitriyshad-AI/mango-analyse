# AMO manual-resolution pipeline, 2026-05-11

This document records the safe post-XLSX workflow for the remaining AMO manual-resolution rows.

## Purpose

The workflow lets an operator review ambiguous AMO contacts in an XLSX file, then converts the reviewed workbook into a validated accepted-only candidate set.

It is fail-closed:

- empty or unresolved XLSX decisions produce `0` live candidates;
- invalid accepted decisions are blocked in `still_blocked.csv`;
- CRM quality gate is required before real-tunnel dry-run;
- live AMO writeback is never executed by this pipeline.

## Main files

- Workbook to fill: `stable_runtime/amo_manual_resolution_20260511_v1/resolution_decisions_manual_template.xlsx`
- One-command pipeline: `scripts/run_amo_resolution_after_xlsx.py`
- Current post-XLSX output: `stable_runtime/amo_manual_resolution_20260511_v2_after_xlsx/`
- Audit pack: `audits/_inbox/amo_manual_resolution_after_xlsx_20260511_v1/`

## Command after XLSX is filled

```bash
cd "/Users/dmitrijfabarisov/Projects/Mango analyse"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. python3 scripts/run_amo_resolution_after_xlsx.py
```

The command does:

1. Converts XLSX to `resolution_decisions_from_xlsx.csv`.
2. Rebuilds manual-resolution outputs from accepted decisions.
3. Writes `decision_qa_summary.json` and `decision_qa_report.md`.
4. Runs CRM writeback quality gate if `resolved_live_candidates_ru.csv` has rows.
5. Writes `next_real_tunnel_dry_run_command.sh`.
6. Refreshes `operator_status_20260511_v1`.
7. Builds a Claude audit pack.

## Current dry run on unfilled workbook

The current unfilled workbook was processed once to verify the workflow:

- Review rows: `16`.
- Accepted rows: `0`.
- Resolved live candidates: `0`.
- Needs human: `14`.
- Already-written review: `2`.
- Quality gate: `skipped_no_resolved_candidates`.
- Live writeback: not executed.

## Safety gates

Accepted rows require:

- `resolution_status` in `accepted_by_manager`, `accepted_by_operator`, `accepted_auto_policy`;
- non-empty `resolved_contact_id`;
- non-empty `resolution_reason`;
- non-empty `resolved_by`.

Additional rules:

- `needs_manager_review_multi_contact` rows are AMO duplicate-merge rows, not normal accepted rows. They require external AMO merge plus post-merge dry-run recheck; accepted reason must contain `post_merge_recheck_approved`.
- Contact id outside `source_amo_contact_ids` requires `allow_contact_id_outside_source=yes` and reason containing `outside_source_approved`.
- `needs_text_quality_review` rows require reason containing `text_quality_approved`.
- Already-written rows require reason containing `refresh_approved`.

## Duplicate merge / post-merge recheck

Manual duplicate merge is allowed only outside this pipeline, inside AMO. For every merged row record the surviving `resolved_contact_id`, list merged IDs in `resolution_notes`, and run `stable_runtime/amo_duplicate_resolution_20260511_v1/next_recheck_command.sh`. If lookup still returns multiple contacts or a different id, the row stays blocked.

## Next stage

After the operator fills the XLSX:

1. Run the one-command pipeline.
2. If `resolved_live_candidates_ru.csv` is non-empty and CRM quality gate passes, run `next_real_tunnel_dry_run_command.sh`.
3. Send the generated audit pack to Claude.
4. Only after clean dry-run and audit, request explicit live writeback approval.

## Updated duplicate-row sequence

Multi-contact duplicate rows are not ordinary accepted rows. `merge completed` без recheck не дает accepted status.

Точная цепочка:

1. Сотрудники вручную склеивают AMO-дубли или подтверждают mismatch в AMO.
2. Release owner запускает `stable_runtime/amo_duplicate_resolution_20260511_v1/next_recheck_command.sh`.
3. Только строки с зеленым row-level recheck получают accepted decision в `resolution_decisions_manual_template.xlsx`.
4. Запускается `scripts/run_amo_resolution_after_xlsx.py`.
5. Если `v2_after_xlsx/resolved_live_candidates_ru.csv` не пустой и CRM quality gate зеленый, запускается `next_real_tunnel_dry_run_command.sh`.
6. Live-stage возможен только после exact-input dry-run, audit, explicit approval и зеленого предыдущего readback.

Для склеенных дублей используйте:

```text
resolution_reason=duplicate_merge_completed_post_merge_recheck_approved
resolution_notes=surviving_contact_id=...; merged_contact_ids=...; merge_done_by=...; merge_done_at=...; recheck_run_dir=...
```
