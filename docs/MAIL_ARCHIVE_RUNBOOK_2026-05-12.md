# Mail Archive Runbook 2026-05-12

## Scope

Read-only REG.RU IMAP archive pilots for local customer-history enrichment.

Hard rules:

- Do not commit raw mail, `.eml`, attachments, extracted text, or archive SQLite files.
- Keep raw artifacts under ignored `_external_handoffs/`.
- Do not write to CRM, Tallanto, Mango, or `stable_runtime`.
- Keep IMAP read-only: `readonly_select=true`, fetch through `BODY.PEEK[]`.
- Pass secrets only through environment variables or local keychain. Do not put passwords in commands, code, reports, or docs.

## Current Controlled Batch Summary

Account label: `regru_cdpofoton_edu`

Verified 30-day archive batches:

- Thematic folder, 30 days, 26 messages.
- `INBOX`, 30 days, 100-message controlled slice.
- `Sent Messages`, 30 days, 100-message controlled slice.
- Contracts folder, 30 days, 5 messages.
- `Sent`, 30 days, 1 message.

Aggregate handoff:

- Source archive count: 5.
- Message count: 232.
- Message kinds: 147 external, 67 internal, 18 service.
- Match classes: 107 strong unique, 25 ambiguous, 15 missing, 85 internal/service.
- Attachments saved as raw bytes only; no attachment content was opened.

Customer-history handoff:

```text
_external_handoffs/mail_archive_2026-05-12/regru_cdpofoton_edu/customer_history_handoff_20260512_30d_controlled/
```

Mango phone bridge preview:

```text
_external_handoffs/mail_archive_2026-05-12/regru_cdpofoton_edu/mango_bridge_preview_20260512_30d_controlled/
```

Bridge preview counts:

- Distinct ready mail candidates: 56.
- Resolved candidates with Mango calls: 12.
- Blocked candidates: 44.
- Blocked because no phone exists for the candidate: 1.
- Blocked because phone identity is not unique: 1.
- Blocked because no Mango phone match exists in current Mango index: 42.
- Mango capture rows read: 21.
- Mango capture rows with normalized phone: 21.
- Mango product call rows with phone parsed from filename: 297.
- Distinct normalized Mango phones indexed: 228.
- Preview call refs written: 20.
- Product call rows read for reference: 297.
- Exact capture-to-product call joins in the current DB: 0.

Expanded Mango phone index preview:

```text
_external_handoffs/mail_archive_2026-05-12/regru_cdpofoton_edu/mango_phone_index_preview_20260513_local_archive/
```

Expanded index counts:

- Product DB phone view distinct phones: 228.
- Local Mango recording filename distinct phones: 400.
- Phone index call refs written: 1 438.
- Recording filename refs written: 1 120.
- Product DB refs written: 318.
- Recording files scanned by name only: 1 168.
- Recording files with parseable filename phone: 1 125.
- Audio files opened: 0.

Expanded Mango bridge preview:

```text
_external_handoffs/mail_archive_2026-05-12/regru_cdpofoton_edu/mango_bridge_preview_20260513_extended_phone_index/
```

Expanded bridge counts:

- Distinct ready mail candidates: 56.
- Resolved candidates with Mango calls: 24.
- Blocked candidates: 32.
- Blocked because no phone exists for the candidate: 1.
- Blocked because phone identity is not unique: 1.
- Blocked because no Mango phone match exists in expanded index: 30.
- Combined distinct normalized Mango phones indexed: 400.
- Preview call refs written: 118.

Phone lift preview for ambiguous/missing mail:

```text
_external_handoffs/mail_archive_2026-05-12/regru_cdpofoton_edu/phone_lift_preview_20260512_30d_controlled/
```

Phone lift preview counts:

- Evaluated manual-review messages: 40.
- Original classes: 25 ambiguous, 15 missing.
- Text files read: 40.
- Extracted phone values seen: 79.
- Distinct phone hashes seen: 20.
- Messages with no phone detected: 7.
- Messages with phones but no Tallanto phone identity match: 33.
- Messages lifted to phone strong-unique: 0.
- Messages remaining manual review: 40.

## Standard Flow

1. Build or refresh Tallanto identity map:

```bash
python3 scripts/mango_office_mail_archive.py identity-map \
  --tallanto-csv _external_handoffs/tallanto_students_export_2026-05-12/Ученики.csv \
  --out-dir _external_handoffs/mail_archive_2026-05-12/regru_edu/identity_map
```

Pass criteria:

- `audit_readiness.pass=true`
- all `sanity_checks` are true
- report contains hashes and aggregate counts
- identity SQLite stays under `_external_handoffs/`

2. Preflight a small batch:

```bash
python3 scripts/mango_office_mail_archive.py preflight \
  --account-label regru_cdpofoton_edu \
  --email <mailbox> \
  --mailbox INBOX \
  --since-days 3 \
  --max-messages 5 \
  --identity-db _external_handoffs/mail_archive_2026-05-12/regru_edu/identity_map/tallanto_email_identity_map.sqlite \
  --out-dir _external_handoffs/mail_archive_2026-05-12/regru_cdpofoton_edu/<batch_id>
```

Pass criteria:

- `preflight_pass=true`
- `blocking_risks=[]`
- password env is present
- output path is git-ignored and not under `stable_runtime`

For a controlled non-pilot window, add `--allow-large-batch`. The approved
guard allows up to 31 days and 250 messages per batch after a verified pilot:

```bash
python3 scripts/mango_office_mail_archive.py preflight \
  --account-label regru_cdpofoton_edu \
  --email <mailbox> \
  --mailbox INBOX \
  --since-days 30 \
  --max-messages 100 \
  --allow-large-batch \
  --identity-db _external_handoffs/mail_archive_2026-05-12/regru_edu/identity_map/tallanto_email_identity_map.sqlite \
  --out-dir _external_handoffs/mail_archive_2026-05-12/regru_cdpofoton_edu/<batch_id>
```

3. Ingest:

```bash
python3 scripts/mango_office_mail_archive.py ingest \
  --account-label regru_cdpofoton_edu \
  --email <mailbox> \
  --mailbox INBOX \
  --since-days 3 \
  --max-messages 5 \
  --out-dir _external_handoffs/mail_archive_2026-05-12/regru_cdpofoton_edu/<batch_id>
```

For larger windows or more than 5 messages, use `--allow-large-batch` only after a verified pilot.
Live ingest also runs preflight before IMAP login and blocks non-git-ignored output paths.

4. Verify:

```bash
python3 scripts/mango_office_mail_archive.py verify-pilot \
  --archive-dir _external_handoffs/mail_archive_2026-05-12/regru_cdpofoton_edu/<batch_id> \
  --expected-max-messages 5
```

Pass criteria:

- `verification_pass=true`
- `blocking_risks=[]`
- DB schema is present
- message count matches raw `.eml` count
- safety block confirms no send/delete/move/write

5. Match:

```bash
python3 scripts/mango_office_mail_archive.py match-report \
  --email <mailbox> \
  --archive-db _external_handoffs/mail_archive_2026-05-12/regru_cdpofoton_edu/<batch_id>/mail_archive.sqlite \
  --identity-db _external_handoffs/mail_archive_2026-05-12/regru_edu/identity_map/tallanto_email_identity_map.sqlite \
  --out-dir _external_handoffs/mail_archive_2026-05-12/regru_cdpofoton_edu/<matching_id>
```

Interpretation:

- `strong_unique`: eligible for read-only customer-history link.
- `ambiguous`: manual review only.
- `missing`: keep unmatched; do not create CRM/Tallanto entities.
- `internal_or_service`: exclude from customer matching.

6. Build handoff:

```bash
python3 scripts/mango_office_mail_archive.py history-handoff \
  --email <mailbox> \
  --identity-db _external_handoffs/mail_archive_2026-05-12/regru_edu/identity_map/tallanto_email_identity_map.sqlite \
  --archive-db <verified_archive_1>/mail_archive.sqlite \
  --archive-db <verified_archive_2>/mail_archive.sqlite \
  --out-dir _external_handoffs/mail_archive_2026-05-12/regru_cdpofoton_edu/customer_history_handoff_<date>
```

7. Build Mango phone index preview:

```bash
python3 scripts/mango_office_mail_archive.py mango-phone-index-preview \
  --product-db _local_archive_mango_api_downloads_20260507/product_appliance/mango_product_appliance.sqlite \
  --recording-root _local_archive_mango_api_downloads_20260507 \
  --out-dir _external_handoffs/mail_archive_2026-05-12/regru_cdpofoton_edu/mango_phone_index_preview_<date>
```

Phone index rules:

- Product DB is opened with SQLite `mode=ro` and `PRAGMA query_only=ON`.
- Local recording roots are scanned by filename only.
- Audio bytes are not opened.
- Source paths and filenames are stored in the index only as SHA256 hashes.
- Raw phone refs are allowed only inside ignored preview SQLite, not in JSON/docs.
- The JSON report contains aggregate counts and safety flags only.

8. Build Mango phone bridge preview:

```bash
python3 scripts/mango_office_mail_archive.py mango-bridge-preview \
  --mail-handoff-db _external_handoffs/mail_archive_2026-05-12/regru_cdpofoton_edu/customer_history_handoff_20260512_30d_controlled/mail_customer_history_handoff.sqlite \
  --identity-db _external_handoffs/mail_archive_2026-05-12/regru_edu/identity_map/tallanto_email_identity_map.sqlite \
  --product-db _local_archive_mango_api_downloads_20260507/product_appliance/mango_product_appliance.sqlite \
  --mango-phone-index-db _external_handoffs/mail_archive_2026-05-12/regru_cdpofoton_edu/mango_phone_index_preview_<date>/mango_phone_index_preview.sqlite \
  --out-dir _external_handoffs/mail_archive_2026-05-12/regru_cdpofoton_edu/mango_bridge_preview_<date>
```

Bridge rules:

- Inputs are opened with SQLite `mode=ro` and `PRAGMA query_only=ON`.
- Source DBs are not attached to the writer connection.
- Only `strong_unique` mail handoff rows participate in resolved bridge rows.
- `ambiguous`, `missing`, and `internal_or_service` mail links stay manual/excluded.
- Phone refs are allowed only inside ignored preview SQLite, not in JSON/docs.
- The JSON report contains aggregate counts and safety flags only.

9. Build phone lift preview for manual-review mail:

```bash
python3 scripts/mango_office_mail_archive.py phone-lift-preview \
  --archive-db <verified_archive_1>/mail_archive.sqlite \
  --archive-db <verified_archive_2>/mail_archive.sqlite \
  --identity-db _external_handoffs/mail_archive_2026-05-12/regru_edu/identity_map/tallanto_email_identity_map.sqlite \
  --out-dir _external_handoffs/mail_archive_2026-05-12/regru_cdpofoton_edu/phone_lift_preview_<date>
```

Phone lift rules:

- Inputs are opened with SQLite `mode=ro` and `PRAGMA query_only=ON`.
- The preview only reads existing `message_matches` rows with `ambiguous` or `missing`.
- It does not recalculate matching, because matching writes to source archive DBs.
- It reads extracted text files only from the same `_external_handoffs` archive directory.
- It does not open attachments or raw `.eml` files.
- It writes phone hashes and candidate refs only inside ignored preview SQLite.
- The JSON report contains aggregate counts and safety flags only.

## Stop Conditions

Stop immediately if any condition appears:

- output is not git-ignored
- output is under `stable_runtime`
- password appears in stdout, JSON, SQLite, file paths, or artifacts
- IMAP operation is not read-only
- any command can send, delete, move, or append mail
- CRM/Tallanto write path is enabled
- raw mail content is about to be copied into docs, chat, or commit
- ambiguous matches are treated as resolved
- Mango bridge reads anything except `strong_unique` handoff rows as resolved
- phone conflicts are promoted to resolved links
- Mango phone index output path is outside ignored `_external_handoffs`
- Mango phone index opens audio bytes instead of scanning filenames only
- bridge output path is outside ignored `_external_handoffs`
- phone lift output path is outside ignored `_external_handoffs`
- extracted text path points outside its source mail archive directory

## Commit-Safe Files

Mail archive code/docs/tests only:

```text
scripts/mango_office_mail_archive.py
src/mango_mvp/productization/mail_archive.py
tests/test_productization_mail_archive.py
docs/MAIL_ARCHIVE_RUNBOOK_2026-05-12.md
```

Do not include unrelated worktree changes in a mail archive commit.

## Scale Decision

Current evidence:

- INBOX has enough activity, but a large share is internal/service.
- `Sent Messages` produces external messages and useful matches, but also the largest ambiguous/missing bucket.
- Thematic and contracts folders produce high strong-unique yield.
- External missing is now material in `Sent Messages` and INBOX, so phone-based enrichment is a useful next investigation after the current email-only bridge.
- Current Mango product DB has only 21 capture rows with phones, so mail-to-Mango coverage is low until broader Mango capture/call phone data is available.
- Current Mango product DB also has 297 product-call filenames with parseable phones; using those raised mail-to-Mango resolved candidates from 2 to 12 in the preview.
- Expanded local Mango filename index raised mail-to-Mango resolved candidates from 12 to 24 in the preview.
- Phone extraction from the current 40 ambiguous/missing mail messages did not produce Tallanto phone matches; keep this as a diagnostic signal, not an automatic resolver.
- Ambiguous matches are present and must remain manual-review only.

Recommended next scale step:

1. Treat the expanded Mango phone index as the current bridge baseline.
2. Continue `Sent Messages` in controlled 100-message slices; stop if ambiguity rises materially.
3. Continue INBOX only in controlled slices because internal/service volume is high.
4. Keep phone lift as preview-only unless a future batch produces strong-unique Tallanto phone matches.
