# Handoff test and security report

Date: 16.08.2026.

## Scope

The handoff branch is based on `eb1c0321da75187e681588d90d56da3638f258ab`. The only new tracked files in this handoff commit are sanitized Markdown specifications and reports under `docs/mango_calls_handoff_20260816/`.

## Required Mango Calls tests

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m pytest -q \
  tests/test_parallel_pipeline.py \
  tests/test_mango_calls_two_processes.py \
  tests/test_publish_current_mango_calls_google.py \
  tests/test_publish_live_mango_calls_google.py
```

Result: `411 passed, 88 warnings` in 17.01 seconds.

## Repository-wide test observation

The unfiltered historical suite cannot collect `tests/test_build_master_exports.py` because that legacy test contains an absolute path to another Mac.

The remaining suite was run with that one file excluded:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m pytest -q \
  --ignore=tests/test_build_master_exports.py
```

Result: `5606 passed, 17 failed, 3 skipped, 10 subtests passed` in 235.45 seconds.

The 17 failures are outside the handoff documentation change. They concern absent historical M4/KB fixtures and absolute paths, three Timeline nightly tests expecting a missing sealed calls DB, one Timeline backup/restore disk-I/O case, and one relocation integration case. They are recorded as baseline debt and must not be described as a green full-suite result.

## Git and secret checks

- `git diff --cached --check`: PASS.
- New documents contain no absolute owner paths, email addresses, private-key markers or known API/token prefixes.
- No `.env`, credential, secret, SQLite, DB, audio, PEM, key or PKCS#12 file is staged.
- The unpushed code history from the nearest published ancestor was scanned for private-key markers and known API/token prefixes: no hits.
- The GitHub repository is publicly readable; therefore full production evidence and runtime snapshots are intentionally excluded from Git.

## Confidential package

Full evidence and the read-only SQLite snapshot are transferred separately as a password-free disk image through the owner's personal Yandex.Disk folder. Integrity is checked with an external SHA-256 file and the internal manifest. Production API credentials and the Google service-account key are not included.
