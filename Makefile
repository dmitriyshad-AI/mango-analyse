.PHONY: test test-smoke audit audit-fast runtime-contract

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
