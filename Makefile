.PHONY: test test-smoke audit audit-fast

test:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q

test-smoke:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m unittest -v tests.test_smoke

audit:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/project_audit.py

audit-fast:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/project_audit.py --skip-tests
