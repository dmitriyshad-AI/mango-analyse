# Risk Review

## Read-only / live safety

- No AMO/CRM/Tallanto writes.
- No message sending.
- No ASR, Resolve+Analyze, RA, LLM, network, subprocess, or API-key path in the new derivation layer.
- CLI dry-run opens SQLite with `open_read_only`.
- `--apply` writes only to the configured local `customer_timeline.sqlite`.
- `stable_runtime/`, runtime DBs, source dumps, CRM/Tallanto sources were not mutated.

## Signal correctness risks

- `paid_no_access` is conservative: it requires positive incoming Tallanto payment and suppresses the signal if matching active access is present.
- Access predicate currently uses normalized fields available after Work B: `tallanto_abonement.visits_left > 0` or matching class access record. If future Tallanto snapshots expose richer access statuses, add fixtures before broad real-data use.
- `hot_lead_silent_7d` depends on deterministic text markers and explicit `as_of`. It does not call a model; therefore recall is intentionally conservative.
- `duplicate_contact` covers explicit open identity/AMO conflicts; it will not infer duplicates from raw phone/name heuristics.

## Idempotency risks checked

- `status/expires_at` are not in `stable_signal_id`.
- Recompute preserves `created_at` and upserts the same `signal_id`.
- Probe repeat after auto-close returned `duplicate: 3`, not inserts.
- Tests cover non-vacuum auto-close for all three signals.

## Residual risks

- Full real-data derive on the large canonical DB was not applied in Work C. This stage delivers logic + CLI; continuous recompute belongs to Work F.
- TTL choices are now code constants: paid_no_access 90d, hot_lead_silent_7d 30d, duplicate_contact 180d. Claude should regrade whether these defaults match operations.
