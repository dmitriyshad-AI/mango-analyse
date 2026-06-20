# Risk Review

## Read/Write Safety

- B2 importer has no subprocess/network/API calls.
- Live read used only external `crm_call.sh` MCP read-only wrapper during the operator probe.
- No AMO/Tallanto/CRM write tools were called.
- No ASR, RA, LLM, message send, stable_runtime mutation, or product runtime DB write was run.
- Repo code writes only to a configured local `customer_timeline.sqlite` when `--apply` is explicitly provided.

## Data Safety

- Raw Tallanto/MCP payloads are not stored in SQLite by the importer.
- The SQLite scrubber now removes Tallanto raw payload key families if they appear in future records.
- The audit pack contains only aggregate counts and command shapes, not raw rows.

## Identity Risk

- Existing `tallanto_student_id` matches are resolved read-only before import.
- Ambiguous matches produce conflicts and no first-match merge.
- No phone/name inference is introduced in B2.

## Known Limits

- `crm_call.sh tallanto_select` limits each call to 50 records.
- B2 does not backfill production timeline DB; real-data apply was only to `/tmp`.
- Full source coverage and production DB application remain blocked on Claude/Dmitry approval.
