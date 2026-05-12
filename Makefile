.PHONY: test test-smoke audit audit-fast runtime-contract runtime-status runtime-artifact-index amo-waiting-pack amo-waiting-post-claude amo-duplicate-staff-tasks amo-post-merge-check amo-after-staff-done

test:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q

test-smoke:
	@printf '%s\n' 'SAFE NOTE: runs tests.test_smoke; includes stable_runtime/rebuild_snapshot.sh only with MANGO_STABLE_SMOKE_ONLY=1.'
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m unittest -v tests.test_smoke

audit:
	@printf '%s\n' 'SAFE NOTE: writes audit artifacts to stable_runtime/project_audit_<timestamp>/ and runs pytest.'
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/project_audit.py

audit-fast:
	@printf '%s\n' 'SAFE NOTE: writes audit artifacts to stable_runtime/project_audit_<timestamp>/ and skips pytest.'
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/project_audit.py --skip-tests

runtime-contract:
	@printf '%s\n' 'SAFE NOTE: read-only; writes stable_runtime/CURRENT_RUNTIME.json.'
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. python3 scripts/mango_office_current_runtime.py --out stable_runtime/CURRENT_RUNTIME.json

runtime-status:
	@printf '%s\n' 'SAFE NOTE: read-only; writes stable_runtime/operator_status_20260511_v1/.'
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. python3 scripts/mango_office_operator_status.py --out-root stable_runtime/operator_status_20260511_v1

runtime-artifact-index:
	@printf '%s\n' 'SAFE NOTE: read-only; indexes stable_runtime artifacts, no delete/move.'
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. python3 scripts/build_runtime_artifact_index.py --project-root . --out-root stable_runtime/runtime_artifact_index_20260511_v1

amo-waiting-pack:
	@printf '%s\n' 'SAFE NOTE: report-only; prepares waiting-work pack, no AMO write.'
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. python3 scripts/run_amo_waiting_autonomous_work.py --project-root . --analysis-date 2026-05-11

amo-waiting-post-claude:
	@printf '%s\n' 'SAFE NOTE: report-only; reads Claude audit result and writes command center, no AMO write.'
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. python3 scripts/run_amo_waiting_post_claude_intake.py

amo-duplicate-staff-tasks:
	@printf '%s\n' 'SAFE NOTE: report-only; builds duplicate merge tasks for staff, no AMO write.'
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. python3 scripts/build_amo_duplicate_staff_tasks.py

amo-post-merge-check:
	@printf '%s\n' 'SAFE NOTE: report-only; checks existing post-merge dry-run reports, no AMO write.'
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. python3 scripts/check_amo_duplicate_post_merge_recheck.py

amo-after-staff-done:
	@printf '%s\n' 'SAFE NOTE: report-only; rebuilds candidates after staff merge confirmation, no AMO write.'
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. python3 scripts/run_amo_duplicate_after_staff_done.py --project-root . --analysis-date 2026-05-11
