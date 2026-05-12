# AMO Duplicate Resolution Runbook, 2026-05-11

Scope: remaining AMO writeback blockers after Stage1/Stage69 are not normal import rows. They are AMO entity-quality problems and must be resolved before live CRM writeback.

## Current Status

Current duplicate-resolution pack:

```text
stable_runtime/amo_duplicate_resolution_20260511_v1/
```

Current counts:

```text
duplicate_contacts_merge_required: 12
contact_id_mismatch_requires_operator: 1
candidate_contact_rows: 27
post_merge_recheck_rows: 13
```

Live writeback remains blocked:

```text
crm_writeback_live_allowed_now=false
queue_ready_rows=0
```

## Main Files

```text
stable_runtime/amo_duplicate_resolution_20260511_v1/duplicate_merge_queue.csv
stable_runtime/amo_duplicate_resolution_20260511_v1/candidate_contacts.csv
stable_runtime/amo_duplicate_resolution_20260511_v1/duplicate_merge_review.xlsx
stable_runtime/amo_duplicate_resolution_20260511_v1/duplicate_merge_review.html
stable_runtime/amo_duplicate_resolution_20260511_v1/post_merge_recheck_input_ru.csv
stable_runtime/amo_duplicate_resolution_20260511_v1/next_recheck_command.sh
```

## Policy

Rows with multiple AMO exact-match contacts must not be treated as accepted manual-resolution rows.

Correct sequence:

1. Open every AMO contact candidate for the phone.
2. Confirm they are duplicates or identify the correct surviving card.
3. Merge duplicates inside AMO manually or assign to the employee responsible for the client context.
4. Employees do not need to fill a manual acceptance workbook for this duplicate block.
5. Run post-merge dry-run recheck.
6. Only if the recheck returns exactly one correct AMO contact ID may the row enter the next dry-run/writeback stage.

The program must not merge AMO contacts automatically. Merge is an external/manual CRM operation.

## Stop Conditions

Do not proceed to live writeback if any are true:

- post-merge dry-run was not executed;
- phone still resolves to multiple AMO contacts;
- live dry-run contact_id differs from the surviving AMO contact_id;
- CRM text quality gate is not green for the exact candidate CSV;
- previous live stage readback is not green;
- operator live approval is missing.

## Recheck Command

After employees merge AMO duplicates, run:

```bash
cd "/Users/dmitrijfabarisov/Projects/Mango analyse"
stable_runtime/amo_duplicate_resolution_20260511_v1/next_recheck_command.sh
```

This command is dry-run only and contains no live-write flags.

## Claude Audit Pack

```text
audits/_inbox/amo_duplicate_resolution_20260511_v1/
```

Claude should verify that duplicate/mismatch rows are fail-closed and cannot bypass post-merge recheck.

## Roles / Кто делает

- Sales ops / ответственный менеджер: подтверждает, что карточки действительно относятся к одному клиенту или семье, выбирает surviving contact_id.
- AMO operator: физически склеивает контакты в amoCRM штатными средствами. Код и AI контакты не склеивают.
- Release owner: запускает post-merge recheck, manual-resolution pipeline, stage dry-run/live gates.
- Claude/Codex: делают read-only аудит и проверки. Их вывод не является approval на live-write.

## Упрощенный процесс после работы сотрудников

Сотрудники самостоятельно склеивают дубли в AMO/Tallanto и сообщают только факт: `готово`.

Дальше код сам:

1. запускает post-merge AMO dry-run по `post_merge_recheck_input_ru.csv`;
2. проверяет, что по каждому телефону остался ровно один AMO contact_id;
3. разрешает surviving contact_id, если он был среди известных candidate contact IDs;
4. пересобирает bounded CSV `post_merge_live_candidates_ru.csv`;
5. требует свежий CRM quality gate и real-tunnel dry-run перед любой live-записью.

Рабочая команда после сообщения сотрудников `готово`:

```bash
cd "/Users/dmitrijfabarisov/Projects/Mango analyse"
stable_runtime/amo_duplicate_resolution_20260511_v1/next_recheck_command.sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. python3 scripts/run_amo_duplicate_after_staff_done.py --project-root . --analysis-date 2026-05-11
```

Если после этого появятся candidate rows, следующие команды уже будут лежать тут:

```text
stable_runtime/amo_duplicate_after_staff_done_20260511_v1/next_quality_gate_command.sh
stable_runtime/amo_duplicate_after_staff_done_20260511_v1/next_real_tunnel_dry_run_command.sh
```

Manual workbook для duplicate rows больше не является обязательным шагом. Он остается только fallback-инструментом для неразрешимых случаев.

## Recheck pass/fail

PASS для строки:

```text
status=dry_run
reason=live_write_not_confirmed
contact_id == resolved_contact_id / surviving contact_id
```

FAIL/BLOCK:

```text
status=skipped или failed
reason=multiple_exact_contacts_in_amo, contact_id_mismatch, contact_not_found_in_amo, runtime preflight error
summary.input не совпадает с post_merge_recheck_input_ru.csv
actual dry_run count меньше expected-dry-run
```

Post-merge recheck не разрешает live-write сам по себе. После него все равно нужны fresh CRM gate, real-tunnel dry-run по точному candidate CSV, audit/approval и readback gate.
