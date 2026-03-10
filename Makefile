.PHONY: test-smoke

test-smoke:
	PYTHONPATH=src python3 -m unittest -v tests.test_smoke
