# Project Cleanup Runbook, 2026-05-11

Scope: build a read-only cleanup manifest for `/Users/dmitrijfabarisov/Projects/Mango analyse`.

This runbook does not authorize deletion, moving, archiving, or quarantine. It only describes how to generate a candidate manifest for human review.

## Command

Default output:

```bash
cd "/Users/dmitrijfabarisov/Projects/Mango analyse"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. python3 scripts/build_project_cleanup_manifest.py
```

Explicit output:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. python3 scripts/build_project_cleanup_manifest.py \
  --project-root . \
  --current-runtime stable_runtime/CURRENT_RUNTIME.json \
  --out-root stable_runtime/project_cleanup_manifest_20260511_v1
```

Deterministic audit cutoff:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. python3 scripts/build_project_cleanup_manifest.py \
  --generated-at 2026-05-11T09:00:00+00:00 \
  --fresh-audit-days 1
```

## Outputs

Default root:

```text
stable_runtime/project_cleanup_manifest_20260511_v1/
```

Files:

```text
manifest.csv
manifest.json
summary.json
```

Required manifest columns:

```text
candidate_path
category
reason
replacement_path
safe_to_quarantine
requires_human_review
```

The CSV also includes `entry_type`, `size_bytes`, and `mtime` for review convenience.

## Safety Contract

The builder is fail-safe by design:

- It scans only the project root, `stable_runtime`, and `audits`.
- It never calls delete, move, quarantine, archive, ASR, R+A, CRM, or Tallanto operations.
- It writes only local report artifacts under `--out-root`.
- It excludes paths pinned by `stable_runtime/CURRENT_RUNTIME.json`.
- It excludes the manifest output directory itself.
- It excludes fresh audit packs using the configured date lookback.

Expected `summary.json` safety block:

```json
{
  "read_only_scan": true,
  "deletes_files": false,
  "moves_files": false,
  "quarantines_files": false,
  "writes_only_report_artifacts": true,
  "destructive_operations_available": false
}
```

## Protected Paths

The builder loads `CURRENT_RUNTIME.json` and protects all resolved paths from its `paths` object.

Typical protected paths include:

```text
stable_runtime/CANONICAL_EXPORT.txt
stable_runtime/CURRENT_RUNTIME.json
stable_runtime/<active strict export>/
stable_runtime/<active canonical master>/
stable_runtime/<active CRM writeback quality gate>/
stable_runtime/<active AMO writeback queue>/
stable_runtime/<active Stage15 gate>/
```

The builder also protects `stable_runtime/amocrm_runtime/` because it contains live writeback and readback evidence.

## Candidate Classes

Common categories:

```text
superseded_strict_export
superseded_canonical_master
superseded_crm_writeback_quality_gate
superseded_amo_writeback_queue
superseded_stage15_gate
superseded_quality_artifact
superseded_status_report
historical_audit_pack
runtime_manual_review_required
runtime_log_or_scratch
root_archive_or_handoff_review
root_binary_document_review
local_os_metadata
```

Interpretation:

- `safe_to_quarantine=true` means the row looks technically safe to move into a quarantine folder after human approval.
- `requires_human_review=true` means no automation should act on the row without an operator decision.
- `safe_to_quarantine=false` means the row is informational or ambiguous and should not be moved by an autonomous cleanup step.

## Review Procedure

1. Generate the manifest.
2. Open `summary.json` and confirm `current_runtime_loaded=true`.
3. Check `protected_runtime_paths` and confirm the current AMO/canonical/writeback chain is protected.
4. Review `manifest.csv` by category.
5. For every row with `safe_to_quarantine=true`, verify `replacement_path` is sufficient evidence.
6. For every row with `requires_human_review=true`, record a separate operator decision before any future quarantine.
7. Do not run any move/delete command from this manifest without a separate approved cleanup implementation and backup/rollback plan.

## Stop Conditions

Stop and do not use the manifest for cleanup planning if any of these are true:

- `current_runtime_loaded=false`.
- A current export, canonical DB, CRM gate, AMO queue, Stage15 gate, or AMO runtime evidence path appears in `manifest.csv`.
- A fresh active audit pack appears in `manifest.csv`.
- `summary.json` safety flags do not match the expected read-only contract.
- The output root is outside the project or points at a production artifact folder.

## Tests

Focused regression:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. python3 -m pytest -q tests/test_productization_project_cleanup_manifest.py
```
