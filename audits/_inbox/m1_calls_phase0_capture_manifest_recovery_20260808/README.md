# M1 Calls Phase 0: capture manifest recovery

## Audit question

Can an interrupted final `capture_manifest.jsonl` write be recovered without
silently accepting unrelated corruption, losing concurrent appends, exposing
the manifest, or allowing a partial Process A result into Process B?

## Baseline and scope

- baseline: `f8faabf1d442261023605ee3285deb3b2a278cf9`;
- branch setup commit: `37e34f39`;
- implementation under audit: the working-tree diff after `37e34f39`;
- code: `src/mango_mvp/productization/capture_staging.py` and
  `src/mango_mvp/customer_timeline/calls_two_processes.py`;
- tests: `tests/test_productization_capture_staging.py` and
  `tests/test_mango_calls_two_processes.py`.

The change does not run ASR, Resolve, Analyze, launchd, cutover, CRM writes, or
modify `stable_runtime`.

## Required invariants

1. Only a final unterminated byte sequence that is demonstrably a prefix of a
   canonical JSON object may be ignored and repaired.
2. Invalid JSON/UTF-8 before the final record and ambiguous final corruption
   fail closed without changing the manifest.
3. Every byte-prefix truncation of a canonical UTF-8 manifest entry is
   recoverable, including a partial multibyte character.
4. A valid final JSON object without `\n` remains valid; append inserts the
   separator.
5. Read, recovery and append use advisory file locks. A live Store rejects
   inode replacement, shrink and non-prefix rewrite; a fresh Store accepts a
   completed offline relocation. A matching recovery-ledger fingerprint is
   required before a peer may replace a previously observed torn tail.
6. Descriptor/path identity and stable signatures are checked around reads and
   writes. FIFO, directory, symlink, disappeared-file and swap-after-open cases
   fail closed without hanging or recreating a previously observed manifest.
   Health and ledger reads also fail red immediately when their lock is busy.
7. Concurrent append calls do not lose records. A legal peer append on the same
   inode is accepted only when the old validated prefix is byte-identical.
8. The file is `0600` before its first data fsync; file data and a newly created
   directory entry are fsynced.
9. Sequential appends through one store reuse the validated snapshot instead
   of reparsing the whole manifest on every append.
10. Recovery survives another process crash through an atomic `0600` ledger.
   Each fingerprint contains SHA and size for both the discarded tail and its
   valid prefix; an incident digest binds the complete fingerprint set. The
   ledger never copies the possibly sensitive broken fragment.
11. Recovery does not depend on a new API event: an empty Mango window repairs
    the tail before polling. The ledger is durable before truncate, so a crash
    between those operations resumes safely; a no-op repeat does not mutate the
    manifest.
12. Ledger refresh and compare-and-ack run under the manifest lock. The durable
   red report carries the incident digest, and acknowledgement compares both
   count and digest. A newer or same-count/different incident cannot be
   acknowledged by an older report.
13. A current or recovered tail makes Process A `partial`, gives the explicit
    stop reason `capture_manifest_tail_incomplete`, leaves the API cursor in
    place and does not create a new sealed drop.
14. Ordinary `partial` runs retain the established RUNBOOK contract: available
    verified calls may proceed to Process B while the overall day stays red.
15. A missing manifest is never treated as a virgin runtime when cursor,
    capture status, recovery ledger or recordings prove prior state.
16. Failed/partial stage status turns red before report-path failures; an `ok`
    stage is published only after its local report is durable.
17. Report identifiers are unique beyond second precision and remain safe for
    the report PII sweep; a clean retry cannot overwrite the red incident
    report.

## Deliberate data-minimization decision

The exact broken fragment is not retained because it can duplicate phone or
other personal data. The valid prefix remains in place, ambiguous corruption
is never deleted, and a successful recovery is recorded in the owner-only run
report plus a durable ledger containing only hashes, byte sizes, count and the
derived incident digest. The ledger is acknowledged only after the red report
is durably written.

## Out of scope for this atomic block

- source-Mac to M1 path relocation;
- before/after transfer inventories;
- capture filename collision repair and atomic audio copy;
- split-host `audio_path` semantics;
- Stage 10 closed-balance verdict and ready-manifest v2.

Those remain STOP until their own implementation and audit blocks land.
