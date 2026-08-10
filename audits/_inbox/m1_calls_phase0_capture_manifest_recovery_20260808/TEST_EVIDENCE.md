# Test evidence

Run from repository root with no live credentials or external writes:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_productization_capture_staging.py tests/test_mango_calls_two_processes.py
147 passed in 5.74s

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_mango_calls_*.py tests/test_productization_capture_staging.py
227 passed in 27.28s

Exact test command from the active task
297 passed in 26.44s

python3 scripts/skills/tz_lint.py tasks/_running/2026-08-07_TZ_m1_calls_runtime_readiness.md
PASS

python3 scripts/preflight.py --root . --tz tasks/_running/2026-08-07_TZ_m1_calls_runtime_readiness.md
PREFLIGHT: OK

git diff --check
clean
```

Synthetic negative controls cover:

- malformed final `{BAD` is rejected and remains byte-identical;
- invalid UTF-8 inside a final object is rejected and remains byte-identical;
- invalid JSON before the last line is rejected;
- schema-valid JSON is still rejected when required entry fields are absent;
- all byte cuts of one canonical UTF-8 entry containing quotes, backslashes,
  escaped controls and multibyte Unicode are accepted only as an incomplete
  final record;
- 24 concurrent append calls retain 24 unique records and a legal peer append
  preserves the validated prefix;
- 20 sequential appends cause one full parse through the same store;
- fresh manifest mode is `0600`;
- inode swap, same-inode shrink, same-size rewrite with restored mtime and
  swap-after-open are rejected without data mutation;
- a fresh Store accepts the relocated inode, while a previously validated Store
  rejects it;
- recovery ledger contains hashes/sizes and a derived incident digest only,
  survives a new store, and compare-and-ack refuses both a newer unreported
  tail and a same-count/different incident;
- a stale reader accepts peer tail repair only when the ledger contains the
  exact tail+prefix fingerprint; replacement without that proof is rejected;
- an empty API window repairs a torn tail without a synthetic entry; a crash
  after durable ledger but before truncate resumes, and a clean no-op repeat is
  byte- and mtime-identical;
- repaired tail is counted, the active capture run stays partial, API cursor
  does not move, no new sealed drop is created, and after durable report+ack a
  clean retry becomes ok;
- a durable report failure leaves the incident unacknowledged;
- failures both before and immediately after acknowledgement preserve the
  original red report and create a distinct failed report;
- FIFO, directories, dangling symlinks, status-marker swaps and duplicate
  recovery fingerprints fail closed without hanging;
- independently held exclusive locks on the manifest or recovery ledger make
  health return red immediately instead of hanging the operator status call;
  this negative control itself runs in a subprocess with a two-second timeout;
- a peer-created manifest is marked as seen and cannot be silently recreated
  after removal;
- a `partial` day cannot become fresh solely because timestamps are recent;
- rapid retries use distinct report paths and preserve the red incident report.

No production audio, API, ASR, Resolve, Analyze, Timeline import, launchd, or
cutover command was executed.
