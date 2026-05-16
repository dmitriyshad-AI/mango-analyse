# Mail Archive Runbook 2026-05-12

## Scope

Read-only REG.RU IMAP archive pilots for local customer-history enrichment.

Hard rules:

- Do not commit raw mail, `.eml`, attachments, extracted text, or archive SQLite files.
- Keep raw artifacts under ignored `_external_handoffs/`.
- Do not write to CRM, Tallanto, Mango, or `stable_runtime`.
- Keep IMAP read-only: `readonly_select=true`, fetch through `BODY.PEEK[]`.
- Pass secrets only through environment variables or local keychain. Do not put passwords in commands, code, reports, or docs.

## Current Primary Controlled Batch Summary

Account label: `regru_edu`

Primary mailbox status:

- REG.RU IMAP login verified through local keychain secret lookup.
- Password value is not stored in code, docs, reports, or shell output.
- `INBOX` pilot passed before the controlled batches.
- Raw artifacts stay under ignored `_external_handoffs/`.

Tallanto identity map:

```text
_external_handoffs/mail_archive_2026-05-12/regru_edu/identity_map/
```

Controlled archive batches:

- `INBOX`, 30 days, 100-message controlled slice.
- `Sent Messages`, 30 days, 100-message controlled slice.
- `INBOX`, 60 to 30 days ago, 100-message controlled slice.
- `Sent Messages`, 60 to 30 days ago, 100-message controlled slice.

Server-side 60-day counters from read-only IMAP search, since 2026-03-13:

- All selectable folders: 7 987 messages.
- `INBOX`: 3 665 messages.
- `Sent`: 3 374 messages.
- `Sent Messages`: 490 messages.

Folder-name note:

- Existing controlled outgoing artifacts were created before mailbox names with
  spaces were quoted inside `ImapLibClient.select`.
- Their server-side counts match the `Sent` folder: 2 046 latest + 1 328 older
  = 3 374.
- Future runs should use `--mailbox Sent` for the high-volume sent folder, or
  rely on the fixed IMAP quoting when intentionally selecting `Sent Messages`.

Full 60-day remaining run:

```text
_external_handoffs/mail_archive_2026-05-12/regru_edu/full_60d_remaining_20260513_v2/
```

Full run scope:

- Planned server-side messages: 7 987.
- Daily windows processed: 350.
- Control message hashes excluded: 400.
- Non-control message source rows inserted or seen: 7 587.
- Unique non-control raw messages in archive: 7 586.
- Attachments saved as raw bytes only: 7 226.
- Extracted text files written: 7 552.
- Errors: 0.
- One new message appeared in the open current-day window after planning and is
  recorded separately, not mixed into this planned run.

Full run matching:

- Message count: 7 586.
- Message kinds: 5 641 external, 405 internal, 1 540 service.
- Match classes: 3 769 strong unique, 1 079 ambiguous, 793 missing,
  1 945 internal/service.
- Distinct matched Tallanto candidates: 1 768.

Full run phone lift:

- Evaluated manual-review messages: 1 872.
- Messages lifted to phone strong-unique: 76.
- Remaining manual review messages: 1 796.
- Text files read: 1 864.
- Text files missing: 8.

Full run Mango bridge:

- Ready mail links: 3 769.
- Manual-review mail links: 1 872.
- Excluded internal/service links: 1 945.
- Distinct ready mail candidates: 1 022.
- Resolved candidates with Mango calls: 149.
- Blocked candidates: 873.
- Preview call refs written: 513.

Full 180-to-60-day archive run:

```text
_external_handoffs/mail_archive_2026-05-12/regru_edu/full_180_to_60d_20260513/
```

Full 180-to-60-day run scope:

- Planned server-side messages: 13 187.
- Daily windows processed: 640.
- Control message hashes excluded: 7 986.
- Messages found and attempted in final run: 13 187.
- Messages excluded as already known by hash: 132.
- Non-control messages inserted or seen in final run: 13 055.
- Unique non-control raw messages in archive after cleanup: 13 055.
- Attachments saved as raw bytes only: 15 125.
- Extracted text files present: 13 015.
- Stale rows from the interrupted first attempt removed from local SQLite: 81.
- Orphan raw `.eml` files from the interrupted first attempt removed locally: 81.
- SHA-256 overlap with the completed 60-day archive after cleanup: 0.
- Errors: 0.

Full 180-to-60-day matching:

- Message count: 13 055.
- Message kinds: 10 848 external, 1 091 internal, 1 116 service.
- Match classes: 8 426 strong unique, 1 541 ambiguous, 881 missing,
  2 207 internal/service.
- Distinct matched Tallanto candidates: 2 152.

Combined 180-day customer-history handoff:

```text
_external_handoffs/mail_archive_2026-05-12/regru_edu/full_180d_combined_20260513/customer_history_handoff_full_180d/
```

Combined 180-day handoff counts:

- Source archive count: 2.
- Message count: 20 641.
- Message kinds: 16 489 external, 1 496 internal, 2 656 service.
- Match classes: 12 195 strong unique, 2 620 ambiguous, 1 674 missing,
  4 152 internal/service.
- Distinct candidate keys: 2 955.

Combined 180-day phone lift:

```text
_external_handoffs/mail_archive_2026-05-12/regru_edu/full_180d_combined_20260513/phone_lift_preview_full_180d/
```

Combined 180-day phone lift counts:

- Evaluated manual-review messages: 4 294.
- Original classes: 2 620 ambiguous, 1 674 missing.
- Messages lifted to phone strong-unique: 117.
- Remaining manual review messages: 4 177.
- Text files read: 4 279.
- Text files missing: 15.

Combined 180-day Mango bridge:

```text
_external_handoffs/mail_archive_2026-05-12/regru_edu/full_180d_combined_20260513/mango_bridge_preview_full_180d_extended_phone_index/
```

Combined 180-day bridge counts:

- Ready mail links: 12 195.
- Manual-review mail links: 4 294.
- Excluded internal/service links: 4 152.
- Distinct ready mail candidates: 2 178.
- Resolved candidates with Mango calls: 174.
- Blocked candidates: 2 004.
- Blocked because no Mango phone match exists in expanded index: 1 947.
- Blocked because phone has multiple candidates: 56.
- Preview call refs written: 644.

Server-history inventory for `edu@kmipt.ru`:

- Total messages on server: 49 917.
- Messages since 365 days: 46 973.
- Messages since 730 days: 49 651.
- Older than 730 days: 266.
- Completed local archive used for reports: 49 480 non-control/processed
  messages, with zero SHA-256 overlap between layers.

Full 365-day combined handoff:

```text
_external_handoffs/mail_archive_2026-05-12/regru_edu/full_365d_combined_20260513/customer_history_handoff_full_365d/
```

Full 365-day counts:

- Source archive count: 3.
- Message count: 46 536.
- Match classes: 27 551 strong unique, 5 725 ambiguous, 3 335 missing,
  9 925 internal/service.
- Distinct candidate keys: 4 810.
- Phone lift strong-unique messages: 205.
- Mango bridge resolved candidates with calls: 191.
- Mango bridge call refs written: 728.

Full 730-day combined handoff:

```text
_external_handoffs/mail_archive_2026-05-12/regru_edu/full_730d_combined_20260513/customer_history_handoff_full_730d/
```

Full 730-day counts:

- Source archive count: 6.
- Message count: 49 214.
- Match classes: 29 520 strong unique, 6 014 ambiguous, 3 593 missing,
  10 087 internal/service.
- Distinct candidate keys: 5 331.
- Phone lift strong-unique messages after matching all 730-day layers: 214.
- Mango bridge resolved candidates with calls: 192.
- Mango bridge call refs written: 732.

Full available mail-history handoff:

```text
_external_handoffs/mail_archive_2026-05-12/regru_edu/full_all_mail_combined_20260513/customer_history_handoff_full_all_mail/
```

Full available history counts:

- Source archive count: 7.
- Message count: 49 480.
- Match classes: 29 711 strong unique, 6 045 ambiguous, 3 599 missing,
  10 125 internal/service.
- Message kinds: 39 355 external, 5 446 internal, 4 679 service.
- Distinct candidate keys: 5 392.
- Phone lift strong-unique messages: 214.
- Mango bridge resolved candidates with calls: 192.
- Mango bridge call refs written: 732.

Safe attachment inventory for full available history:

```text
_external_handoffs/mail_archive_2026-05-12/regru_edu/full_all_mail_combined_20260513/attachment_inventory_safe/
```

Attachment inventory counts:

- Attachments: 56 951.
- Messages with attachments: 26 506.
- Total attachment bytes: 27 130 520 990.
- Largest attachment bytes: 24 362 030.
- Size buckets: 333 zero-byte, 22 849 under 100KB, 28 371 from 100KB to
  1MB, 4 580 from 1MB to 5MB, 790 from 5MB to 20MB, 28 at or above 20MB.
- Top safe extension buckets: pdf 22 254, jpg 13 955, png 13 247,
  jpeg 3 133, docx 1 742, gif 998, doc 722.
- Container/suspicious buckets observed by extension only: 32 zip, 4 rar,
  1 7z.
- Attachment contents were not opened, executed, OCRed, extracted, or sent to
  ASR/R+A.

Attachment analysis plan:

- First pass: allowlist only `pdf`, `docx`, `xlsx`, `csv`, `txt`, `png`,
  `jpg/jpeg`, `webp`; write derived text next to the source `sha256`.
- Do not open or unpack archives, executables, scripts, or macro formats
  (`docm`, `xlsm`, `pptm`) without a separate explicit decision.
- Cap the first parser pass at 20MB per file; leave larger files in a manual
  queue.
- Keep raw files immutable. Store derived output under ignored handoff
  directories and link it by message/attachment `sha256`.
- Do not run ASR/R+A in this mail branch.

Latest `INBOX` batch counts:

- Messages found since 2026-04-12: 2 177.
- Messages attempted/inserted: 100.
- Raw `.eml` written: 100.
- Extracted text files written: 98.
- Attachments saved as raw bytes only: 52.
- Verification pass: true.
- Matching: 30 strong unique, 0 ambiguous, 14 missing, 56 internal/service.

Latest `Sent Messages` batch counts:

- Messages found since 2026-04-12: 2 046.
- Messages attempted/inserted: 100.
- Raw `.eml` written: 100.
- Extracted text files written: 100.
- Attachments saved as raw bytes only: 118.
- Verification pass: true.
- Matching: 71 strong unique, 8 ambiguous, 21 missing, 0 internal/service.

Older `INBOX` batch counts:

- Window: 2026-03-13 through 2026-04-11.
- Messages found in window: 1 488.
- Messages attempted/inserted: 100.
- Raw `.eml` written: 100.
- Extracted text files written: 99.
- Attachments saved as raw bytes only: 89.
- Verification pass: true.
- Matching: 40 strong unique, 15 ambiguous, 15 missing, 30 internal/service.

Older `Sent Messages` batch counts:

- Window: 2026-03-13 through 2026-04-11.
- Messages found in window: 1 328.
- Messages attempted/inserted: 100.
- Raw `.eml` written: 100.
- Extracted text files written: 100.
- Attachments saved as raw bytes only: 131.
- Verification pass: true.
- Matching: 66 strong unique, 31 ambiguous, 1 missing, 2 internal/service.

Current 60-day customer-history handoff:

```text
_external_handoffs/mail_archive_2026-05-12/regru_edu/customer_history_handoff_20260513_60d_controlled/
```

Current 60-day handoff counts:

- Source archive count: 4.
- Message count: 400.
- Message kinds: 312 external, 20 internal, 68 service.
- Match classes: 207 strong unique, 54 ambiguous, 51 missing, 88 internal/service.
- Distinct candidate keys: 289.

Previous 30-day customer-history handoff:

```text
_external_handoffs/mail_archive_2026-05-12/regru_edu/customer_history_handoff_20260513_inbox_sent_30d_controlled/
```

Handoff counts:

- Source archive count: 2.
- Message count: 200.
- Message kinds: 144 external, 56 service.
- Match classes: 101 strong unique, 8 ambiguous, 35 missing, 56 internal/service.
- Distinct candidate keys: 79.

Expanded Mango phone index preview:

```text
_external_handoffs/mail_archive_2026-05-12/regru_edu/mango_phone_index_preview_20260513_local_archive/
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

Current 60-day phone lift preview for ambiguous/missing mail:

```text
_external_handoffs/mail_archive_2026-05-12/regru_edu/phone_lift_preview_20260513_60d_controlled/
```

Current 60-day phone lift counts:

- Evaluated manual-review messages: 105.
- Original classes: 54 ambiguous, 51 missing.
- Messages lifted to phone strong-unique: 3.
- Messages remaining manual review: 102.
- Text files read: 104.
- Text files missing: 1.

Previous 30-day phone lift preview:

```text
_external_handoffs/mail_archive_2026-05-12/regru_edu/phone_lift_preview_20260513_inbox_sent_30d/
```

Phone lift counts:

- Evaluated manual-review messages: 43.
- Original classes: 8 ambiguous, 35 missing.
- Messages lifted to phone strong-unique: 2.
- Messages remaining manual review: 41.

Current 60-day Mango bridge preview:

```text
_external_handoffs/mail_archive_2026-05-12/regru_edu/mango_bridge_preview_20260513_60d_controlled_extended_phone_index/
```

Current 60-day bridge counts:

- Ready mail links: 207.
- Manual-review mail links: 105.
- Excluded service/internal links: 88.
- Distinct ready mail candidates: 115.
- Resolved candidates with Mango calls: 34.
- Blocked candidates: 81.
- Blocked because no Mango phone match exists in expanded index: 80.
- Blocked because phone has multiple candidates: 1.
- Preview call refs written: 113.

Previous 30-day Mango bridge preview:

```text
_external_handoffs/mail_archive_2026-05-12/regru_edu/mango_bridge_preview_20260513_inbox_sent_30d_extended_phone_index/
```

Bridge counts:

- Ready mail links: 101.
- Manual-review mail links: 43.
- Excluded service/internal links: 56.
- Distinct ready mail candidates: 65.
- Resolved candidates with Mango calls: 30.
- Blocked candidates: 35.
- Blocked because no Mango phone match exists in expanded index: 35.
- Preview call refs written: 101.

## Previous Secondary Controlled Batch Summary

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
guard allows up to 31 days and 500 messages per batch after a verified pilot:

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

For the next older monthly slice, use `--before-days` so the batch does not
repeat the latest `SINCE` results. Example: `--since-days 60 --before-days 30`
means the 30-day window from 60 to 30 days ago.

Prepared next `Sent Messages` preflight:

```text
_external_handoffs/mail_archive_2026-05-12/regru_cdpofoton_edu/folder_sent_messages_20260513_60_to_30d_100msg/
```

Current status after ingest:

- Window: 60 to 30 days ago.
- Max messages: 100.
- Output path: git-ignored and outside `stable_runtime`.
- Identity DB: present.
- Preflight blocking risks: none.
- Messages found/attempted/inserted: 7.
- Raw `.eml` written: 7.
- Extracted text files written: 7.
- Attachments saved as raw bytes only: 3.
- Verification pass: true.
- Matching: 0 strong unique, 0 ambiguous, 2 missing, 5 internal/service.

Updated consolidated handoff with this older slice:

```text
_external_handoffs/mail_archive_2026-05-12/regru_cdpofoton_edu/customer_history_handoff_20260513_30d_plus_sent_60_to_30d/
```

Updated handoff counts:

- Source archive count: 6.
- Message count: 239.
- Message kinds: 149 external, 72 internal, 18 service.
- Match classes: 107 strong unique, 25 ambiguous, 17 missing, 90 internal/service.

Updated bridge with expanded Mango phone index:

```text
_external_handoffs/mail_archive_2026-05-12/regru_cdpofoton_edu/mango_bridge_preview_20260513_30d_plus_sent_60_to_30d_extended_phone_index/
```

Updated bridge counts:

- Distinct ready mail candidates: 56.
- Resolved candidates with Mango calls: 24.
- Blocked candidates: 32.
- Blocked because no Mango phone match exists in expanded index: 30.

Updated phone lift preview:

```text
_external_handoffs/mail_archive_2026-05-12/regru_cdpofoton_edu/phone_lift_preview_20260513_30d_plus_sent_60_to_30d/
```

Updated phone lift counts:

- Evaluated manual-review messages: 42.
- Original classes: 25 ambiguous, 17 missing.
- Messages lifted to phone strong-unique: 0.
- Messages remaining manual review: 42.

3. Ingest:

```bash
python3 scripts/mango_office_mail_archive.py ingest \
  --account-label regru_cdpofoton_edu \
  --email <mailbox> \
  --mailbox INBOX \
  --since-days 3 \
  --before-days <optional_days_ago> \
  --max-messages 5 \
  --out-dir _external_handoffs/mail_archive_2026-05-12/regru_cdpofoton_edu/<batch_id>
```

For larger windows or more than 5 messages, use `--allow-large-batch` only after a verified pilot.
Live ingest also runs preflight before IMAP login and blocks non-git-ignored output paths.
For older monthly slices, keep `since_days - before_days <= 31`.

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

10. Build attachment parse plan:

```bash
python3 scripts/mango_office_mail_archive.py attachment-parse-plan \
  --archive-db <verified_archive_1>/mail_archive.sqlite \
  --archive-db <verified_archive_2>/mail_archive.sqlite \
  --out-dir _external_handoffs/mail_archive_2026-05-12/regru_edu/full_all_mail_combined_20260513/attachment_parse_plan_stage1
```

Attachment parse-plan rules:

- Inputs are opened with SQLite `mode=ro` and `PRAGMA query_only=ON`.
- The command reads only the archive `attachments` metadata table.
- It does not open attachments, raw `.eml`, or extracted text files.
- It does not extract archives, execute files, run OCR, run ASR/R+A, or write CRM/Tallanto.
- Raw attachment filenames and raw attachment paths are not written to the plan DB or JSON report.
- The plan stores message hashes, attachment hashes, filename hashes, extension, declared content type, size, action, and risk reasons.
- `parse_later` means eligible for a future controlled parser stage, not parsed in this stage.
- `manual_review` and `blocked` stay out of automatic parsing.

Stage 1 output, built on 2026-05-13:

- Output dir: `_external_handoffs/mail_archive_2026-05-12/regru_edu/full_all_mail_combined_20260513/attachment_parse_plan_stage1`
- Source archive DBs: 7
- Attachments planned: 56,951
- `parse_later`: 55,482
- `manual_review`: 1,068
- `blocked`: 401
- Plan DB: `mail_attachment_parse_plan.sqlite`
- Safe aggregate report: `mail_attachment_parse_plan_report.json`

11. Extract text from safe attachment formats:

```bash
python3 scripts/mango_office_mail_archive.py attachment-text-extract \
  --archive-db <verified_archive_1>/mail_archive.sqlite \
  --archive-db <verified_archive_2>/mail_archive.sqlite \
  --parse-plan-db _external_handoffs/mail_archive_2026-05-12/regru_edu/full_all_mail_combined_20260513/attachment_parse_plan_stage1/mail_attachment_parse_plan.sqlite \
  --out-dir _external_handoffs/mail_archive_2026-05-12/regru_edu/full_all_mail_combined_20260513/attachment_text_extract_stage2
```

Attachment text-extract rules:

- Inputs are opened with SQLite `mode=ro` and `PRAGMA query_only=ON`.
- Only `attachment_parse_plan.action = 'parse_later'` rows are eligible.
- Stage 2 supports only `.txt`, `.csv`, `.docx`, and `.xlsx`.
- PDF, images, archives, legacy Office, macro Office, HTML, ICS, and executable/script formats are not parsed in this stage.
- Attachment paths are accepted only from the source archive `attachments` directory, under `_external_handoffs`, outside `stable_runtime`, with the expected hash-based `.bin` name.
- Office files are checked before parsing for zip path abuse, macro payloads, external relationships, external links, connections, ActiveX, and embedded objects.
- XLSX is opened read-only, data-only, with external links disabled; formulas and hyperlinks are noted as warnings and not followed.
- The SQLite output stores hashes, status, parser, counts, warnings, and derived text paths only.
- The JSON report contains aggregate counts and safety flags only; it does not contain extracted text, raw filenames, raw attachment paths, email addresses, phones, or passwords.
- Derived text files may contain personal data and must stay under ignored `_external_handoffs`.

Stage 2 output, built on 2026-05-13:

- Output dir: `_external_handoffs/mail_archive_2026-05-12/regru_edu/full_all_mail_combined_20260513/attachment_text_extract_stage2`
- Parse-plan queue: 55,482
- Stage-supported queue: 1,798
- Skipped for later stages: 53,684
- Extracted: 1,763
- Blocked by safety: 19
- Empty text: 14
- Parse error: 1
- Stage size limit exceeded: 1
- Derived text chars: 3,202,452
- Extract DB: `mail_attachment_text_extract.sqlite`
- Safe aggregate report: `mail_attachment_text_extract_report.json`
- Derived text dir: `attachment_text/`

12. Extract text from safe PDF attachments:

```bash
python3 scripts/mango_office_mail_archive.py attachment-pdf-extract \
  --archive-db <verified_archive_1>/mail_archive.sqlite \
  --archive-db <verified_archive_2>/mail_archive.sqlite \
  --parse-plan-db _external_handoffs/mail_archive_2026-05-12/regru_edu/full_all_mail_combined_20260513/attachment_parse_plan_stage1/mail_attachment_parse_plan.sqlite \
  --out-dir _external_handoffs/mail_archive_2026-05-12/regru_edu/full_all_mail_combined_20260513/attachment_pdf_extract_stage3
```

Attachment PDF-extract rules:

- Inputs are opened with SQLite `mode=ro` and `PRAGMA query_only=ON`.
- Only `attachment_parse_plan.action = 'parse_later'` rows with `.pdf` are eligible.
- PDF parsing is a separate stage from Office/text parsing.
- The parser extracts embedded text only; it does not render PDF pages, extract images, run OCR, execute actions, follow links, or read raw `.eml`.
- Attachment paths are accepted only from the source archive `attachments` directory, under `_external_handoffs`, outside `stable_runtime`, with the expected hash-based `.bin` name.
- Files are blocked before/while parsing if they declare JavaScript, open/additional actions, embedded files, remote links, AcroForm/XFA, encrypted content, or too many pages.
- Limits: 20 MB per attachment, 10 PDF pages, 100,000 text chars per attachment, 25,000 chars per page, 10 seconds per PDF.
- The SQLite output stores hashes, status, parser, page counts, warnings, and derived text paths only.
- The JSON report contains aggregate counts and safety flags only; it does not contain extracted text, raw filenames, raw attachment paths, email addresses, phones, or passwords.
- Derived PDF text files may contain personal data and must stay under ignored `_external_handoffs`.

Stage 3 output, built on 2026-05-13:

- Output dir: `_external_handoffs/mail_archive_2026-05-12/regru_edu/full_all_mail_combined_20260513/attachment_pdf_extract_stage3`
- Parse-plan queue: 55,482
- Stage-supported queue: 22,217
- Skipped for other stages: 33,265
- Extracted: 5,571
- Blocked by safety: 13,663
- Empty text: 2,979
- Parse errors: 4
- PDF pages processed: 17,174
- Derived text chars: 20,772,971
- Derived text files: 5,571
- Re-run check: same status counts, `derived_text_files_written = 0`
- Extract DB: `mail_attachment_pdf_extract.sqlite`
- Safe aggregate report: `mail_attachment_pdf_extract_report.json`
- Derived text dir: `attachment_pdf_text/`

13. Plan image OCR candidates safely:

```bash
python3 scripts/mango_office_mail_archive.py attachment-image-ocr-plan \
  --archive-db <verified_archive_1>/mail_archive.sqlite \
  --archive-db <verified_archive_2>/mail_archive.sqlite \
  --parse-plan-db _external_handoffs/mail_archive_2026-05-12/regru_edu/full_all_mail_combined_20260513/attachment_parse_plan_stage1/mail_attachment_parse_plan.sqlite \
  --out-dir _external_handoffs/mail_archive_2026-05-12/regru_edu/full_all_mail_combined_20260513/attachment_image_ocr_plan_stage4
```

Attachment image OCR-plan rules:

- Inputs are opened with SQLite `mode=ro` and `PRAGMA query_only=ON`.
- Only `attachment_parse_plan.action = 'parse_later'` rows with `.png`, `.jpg`, `.jpeg`, or `.webp` are eligible.
- The default full-run mode is metadata-only: it does not read attachment bytes, inspect image dimensions, parse EXIF, decode images, write thumbnails, write images, or run OCR.
- OCR is disabled in this stage. `ocr_status = 'disabled'` means the attachment is a future OCR candidate, not an OCR result.
- The optional `--inspect-headers` flag is for small synthetic/pilot checks only. It may read attachment bytes for image headers, but still must not decode full images, extract EXIF, write thumbnails, write images, or run OCR.
- The SQLite output stores hashes, status, extension, declared content type, size, OCR status, and safe aggregate image metadata only.
- The JSON report contains aggregate counts and safety flags only; it does not contain raw filenames, raw attachment paths, raw image content, raw OCR text, EXIF, email addresses, phones, or passwords.

Stage 4 output, built on 2026-05-13:

- Output dir: `_external_handoffs/mail_archive_2026-05-12/regru_edu/full_all_mail_combined_20260513/attachment_image_ocr_plan_stage4`
- Parse-plan queue: 55,482
- Stage-supported queue: 30,074
- Skipped for other stages: 25,408
- Planned OCR-disabled candidates: 30,074
- Blocked by safety: 0
- Parse errors: 0
- Attachment bytes read: 0
- Derived image files: 0
- Thumbnails: 0
- OCR text files: 0
- Top image extensions: `.jpg` 13,666, `.png` 13,224, `.jpeg` 3,129, `.webp` 55
- Re-run check: same status counts and row count, no duplicate accumulation
- Plan DB: `mail_attachment_image_ocr_plan.sqlite`
- Safe aggregate report: `mail_attachment_image_ocr_plan_report.json`

14. Build unified attachment text index:

```bash
python3 scripts/mango_office_mail_archive.py attachment-text-index \
  --text-extract-db _external_handoffs/mail_archive_2026-05-12/regru_edu/full_all_mail_combined_20260513/attachment_text_extract_stage2/mail_attachment_text_extract.sqlite \
  --pdf-extract-db _external_handoffs/mail_archive_2026-05-12/regru_edu/full_all_mail_combined_20260513/attachment_pdf_extract_stage3/mail_attachment_pdf_extract.sqlite \
  --image-ocr-plan-db _external_handoffs/mail_archive_2026-05-12/regru_edu/full_all_mail_combined_20260513/attachment_image_ocr_plan_stage4/mail_attachment_image_ocr_plan.sqlite \
  --parse-plan-db _external_handoffs/mail_archive_2026-05-12/regru_edu/full_all_mail_combined_20260513/attachment_parse_plan_stage1/mail_attachment_parse_plan.sqlite \
  --out-dir _external_handoffs/mail_archive_2026-05-12/regru_edu/full_all_mail_combined_20260513/attachment_text_index_stage5
```

Attachment text-index rules:

- Inputs are opened with SQLite `mode=ro` and `PRAGMA query_only=ON`.
- This stage is metadata-only. It reads Stage 2/3/4 SQLite rows, but does not read derived `.txt` content, raw `.eml`, raw attachment bytes, images, PDFs, or OCR output.
- It unifies Stage 2 text/Office metadata, Stage 3 PDF-text metadata, and Stage 4 image OCR-pending metadata.
- The SQLite output stores hashes, extension, content type, source status, normalized text availability status, parser name, counts, warnings, and hashed derived-text path only.
- It does not store raw filenames, raw attachment paths, raw derived text paths, extracted text content, OCR text, email addresses, phones, or passwords in the JSON report.
- `text_status = 'available'` means Stage 2/3 already produced extracted text. `text_status = 'ocr_pending'` means Stage 4 identified a future OCR candidate but no OCR text exists.

Stage 5 output, built on 2026-05-13:

- Output dir: `_external_handoffs/mail_archive_2026-05-12/regru_edu/full_all_mail_combined_20260513/attachment_text_index_stage5`
- Source DBs: 3
- Parse-plan queue: 55,482
- Covered source rows: 54,089
- Available text rows: 7,334
- OCR-pending image rows: 30,074
- Needs-review or unavailable rows: 46,755
- Parse-later rows without Stage 5 source: 1,393
- Duplicate attachment keys: 0
- Text status counts: available 7,334, blocked 13,682, empty text 2,993, OCR pending 30,074, parse error 5, skipped 1
- Source stage counts: text extract 1,798, PDF extract 22,217, image OCR plan 30,074
- Derived text files read: 0
- Attachment bytes read: 0
- Raw text chars written: 0
- Re-run check: same status counts and row count, no duplicate accumulation
- Index DB: `mail_attachment_text_index.sqlite`
- Safe aggregate report: `mail_attachment_text_index_report.json`

15. Build Stage 6 gap and gated OCR pilot plan:

```bash
python3 scripts/mango_office_mail_archive.py attachment-stage6-plan \
  --parse-plan-db _external_handoffs/mail_archive_2026-05-12/regru_edu/full_all_mail_combined_20260513/attachment_parse_plan_stage1/mail_attachment_parse_plan.sqlite \
  --text-index-db _external_handoffs/mail_archive_2026-05-12/regru_edu/full_all_mail_combined_20260513/attachment_text_index_stage5/mail_attachment_text_index.sqlite \
  --out-dir _external_handoffs/mail_archive_2026-05-12/regru_edu/full_all_mail_combined_20260513/attachment_stage6_plan \
  --ocr-pilot-limit 15 \
  --max-pilot-attachment-bytes 5000000
```

Attachment Stage 6 rules:

- Inputs are opened with SQLite `mode=ro` and `PRAGMA query_only=ON`.
- This stage is metadata-only. It reads Stage 1 parse-plan metadata and Stage 5 index metadata, but does not read derived `.txt` content, raw `.eml`, raw attachment bytes, images, PDFs, or OCR output.
- It classifies `parse_later` rows that have no Stage 2/3/4 source and keeps them as manual-review parser backlog.
- It chooses a tiny deterministic OCR pilot sample from Stage 4 `ocr_pending` rows only.
- OCR remains disabled. No image decoding, EXIF extraction, thumbnails, image writes, or OCR text writes are allowed.
- The SQLite output stores hashes, extension, content type, size, gap class, recommended next action, pilot status, and pilot rank only.
- It does not store raw filenames, raw attachment paths, raw derived text paths, extracted text content, OCR text, email addresses, phones, or passwords in the JSON report.

Stage 6 output, built on 2026-05-13:

- Output dir: `_external_handoffs/mail_archive_2026-05-12/regru_edu/full_all_mail_combined_20260513/attachment_stage6_plan`
- Parse-plan queue: 55,482
- Gap rows written: 1,393
- Gap extensions: `.gif` 998, `.heic` 282, `.ics` 64, `.html` 28, `.htm` 21
- Gap classes: unsupported image format 1,280, calendar invite format 64, unsafe markup format 49
- Recommended actions: future/manual image parser 1,280, calendar parser review 64, HTML parser review 49
- OCR candidates: 30,074
- OCR eligible under 5 MB: 29,782
- OCR selected for pilot: 15
- OCR selected by extension: `.jpeg` 4, `.jpg` 4, `.png` 4, `.webp` 3
- OCR deferred: 29,767
- OCR excluded by pilot guards: 292
- Derived text files read: 0
- Attachment bytes read: 0
- OCR text files written: 0
- Images/thumbnails written: 0
- Re-run check: same status counts and row count, no duplicate accumulation
- Plan DB: `mail_attachment_stage6_plan.sqlite`
- Safe aggregate report: `mail_attachment_stage6_plan_report.json`

16. Verify selected OCR pilot attachments before OCR:

```bash
python3 scripts/mango_office_mail_archive.py attachment-ocr-preflight \
  --archive-db <verified_archive_1>/mail_archive.sqlite \
  --archive-db <verified_archive_2>/mail_archive.sqlite \
  --stage6-plan-db _external_handoffs/mail_archive_2026-05-12/regru_edu/full_all_mail_combined_20260513/attachment_stage6_plan/mail_attachment_stage6_plan.sqlite \
  --out-dir _external_handoffs/mail_archive_2026-05-12/regru_edu/full_all_mail_combined_20260513/attachment_ocr_preflight \
  --max-candidates 15 \
  --max-attachment-bytes 5000000
```

Attachment OCR preflight rules:

- Inputs are opened with SQLite `mode=ro` and `PRAGMA query_only=ON`.
- This preflight reads bytes only for selected Stage 6 OCR pilot candidates and only to verify file size/path/hash.
- It does not decode images, extract EXIF, write thumbnails, write image files, run OCR, read raw `.eml`, read derived text, or write OCR text.
- Attachment paths are accepted only from the source archive `attachments` directory, under `_external_handoffs`, outside `stable_runtime`, with the expected hash-based `.bin` name.
- The SQLite output stores hashes, input DB path hashes, extension, content type, size, pilot rank, verification status, and bytes-read counts only.
- It does not store raw filenames, raw attachment paths, raw image content, raw OCR text, EXIF, email addresses, phones, or passwords in the JSON report.

OCR preflight output, built on 2026-05-14:

- Output dir: `_external_handoffs/mail_archive_2026-05-12/regru_edu/full_all_mail_combined_20260513/attachment_ocr_preflight`
- Source archive DBs: 7
- Selected candidates checked: 15
- Verified by sha256: 15
- Skipped: 0
- Blocked by safety: 0
- Attachment bytes read: 78,040
- OCR text files written: 0
- Images/thumbnails written: 0
- Extension counts: `.jpeg` 4, `.jpg` 4, `.png` 4, `.webp` 3
- Re-run check: same status counts and row count, no duplicate accumulation
- Preflight DB: `mail_attachment_ocr_preflight.sqlite`
- Safe aggregate report: `mail_attachment_ocr_preflight_report.json`

17. Run verified OCR pilot:

```bash
python3 scripts/mango_office_mail_archive.py attachment-ocr-pilot \
  --archive-db <verified_archive_1>/mail_archive.sqlite \
  --archive-db <verified_archive_2>/mail_archive.sqlite \
  --ocr-preflight-db _external_handoffs/mail_archive_2026-05-12/regru_edu/full_all_mail_combined_20260513/attachment_ocr_preflight/mail_attachment_ocr_preflight.sqlite \
  --out-dir _external_handoffs/mail_archive_2026-05-12/regru_edu/full_all_mail_combined_20260513/attachment_ocr_pilot \
  --max-candidates 15 \
  --max-attachment-bytes 5000000 \
  --languages rus+eng \
  --psm 6 \
  --tesseract-timeout-seconds 30
```

Attachment OCR pilot rules:

- Inputs are opened with SQLite `mode=ro` and `PRAGMA query_only=ON`.
- OCR runs only for selected preflight rows with `verification_status = 'verified'`.
- The command reads bytes only for verified OCR pilot candidates and re-checks sha256 immediately before sending the file to Tesseract.
- Attachment paths are accepted only from the source archive `attachments` directory, under `_external_handoffs`, outside `stable_runtime`, with the expected hash-based `.bin` name.
- OCR uses the local Tesseract CLI with `rus+eng`; no image files, thumbnails, EXIF, PDFs, archive extraction, raw `.eml`, CRM, Tallanto, ASR, or R+A paths are used.
- The SQLite output stores hashes, input DB path hashes, extension, content type, size, OCR status, text hashes, character counts, and hashed derived text path only.
- The JSON report contains aggregate counts and safety flags only; it does not contain raw OCR text, raw filenames, raw attachment paths, email addresses, phones, or passwords.
- Derived OCR text files may contain personal data and must stay under ignored `_external_handoffs`.

OCR pilot output, built on 2026-05-14:

- Output dir: `_external_handoffs/mail_archive_2026-05-12/regru_edu/full_all_mail_combined_20260513/attachment_ocr_pilot`
- Source archive DBs: 7
- Selected verified candidates processed: 15
- Extracted text: 4
- Empty text: 11
- OCR errors: 0
- Blocked by safety: 0
- Attachment bytes submitted to OCR: 78,040
- OCR text characters: 71
- OCR text files present: 4
- Images/thumbnails written: 0
- Extension counts: `.jpeg` 4, `.jpg` 4, `.png` 4, `.webp` 3
- Re-run check: same status counts and row count; second run wrote 0 new OCR text files
- Leak check: JSON and SQLite contain no passwords, email values, or raw OCR text
- OCR DB: `mail_attachment_ocr_pilot.sqlite`
- Safe aggregate report: `mail_attachment_ocr_pilot_report.json`

18. Run controlled parallel OCR batch:

```bash
python3 scripts/mango_office_mail_archive.py attachment-stage6-plan \
  --parse-plan-db _external_handoffs/mail_archive_2026-05-12/regru_edu/full_all_mail_combined_20260513/attachment_parse_plan_stage1/mail_attachment_parse_plan.sqlite \
  --text-index-db _external_handoffs/mail_archive_2026-05-12/regru_edu/full_all_mail_combined_20260513/attachment_text_index_stage5/mail_attachment_text_index.sqlite \
  --out-dir _external_handoffs/mail_archive_2026-05-12/regru_edu/full_all_mail_combined_20260513/attachment_ocr_batch_1100_stage6_plan \
  --ocr-pilot-limit 1100 \
  --max-pilot-attachment-bytes 5000000

python3 scripts/mango_office_mail_archive.py attachment-ocr-preflight \
  --archive-db <verified_archive_1>/mail_archive.sqlite \
  --archive-db <verified_archive_2>/mail_archive.sqlite \
  --stage6-plan-db _external_handoffs/mail_archive_2026-05-12/regru_edu/full_all_mail_combined_20260513/attachment_ocr_batch_1100_stage6_plan/mail_attachment_stage6_plan.sqlite \
  --out-dir _external_handoffs/mail_archive_2026-05-12/regru_edu/full_all_mail_combined_20260513/attachment_ocr_batch_1100_preflight \
  --max-candidates 1100 \
  --max-attachment-bytes 5000000

python3 scripts/mango_office_mail_archive.py attachment-ocr-pilot \
  --archive-db <verified_archive_1>/mail_archive.sqlite \
  --archive-db <verified_archive_2>/mail_archive.sqlite \
  --ocr-preflight-db _external_handoffs/mail_archive_2026-05-12/regru_edu/full_all_mail_combined_20260513/attachment_ocr_batch_1100_preflight/mail_attachment_ocr_preflight.sqlite \
  --out-dir _external_handoffs/mail_archive_2026-05-12/regru_edu/full_all_mail_combined_20260513/attachment_ocr_batch_1100 \
  --max-candidates 1100 \
  --max-attachment-bytes 5000000 \
  --languages rus+eng \
  --psm 6 \
  --tesseract-timeout-seconds 30 \
  --workers 11 \
  --tesseract-thread-limit 1
```

Parallel OCR batch rules:

- Use `--workers` for parallel Tesseract processes.
- Keep `--tesseract-thread-limit 1` for high worker counts so each Tesseract process uses one internal OpenMP thread.
- All OCR pilot safety rules still apply: only verified preflight candidates, sha256 re-check immediately before Tesseract, ignored output directory, no raw OCR text in JSON/SQLite.
- The current command is safe to re-run from a data-integrity perspective, but it still re-runs Tesseract over the selected candidates. A future nightly runner should add a persistent skip-already-processed queue to avoid wasted CPU.

Controlled 1100-image OCR batch output, built on 2026-05-14:

- Stage 6 selected candidates: 1,100
- Selected by extension: `.jpeg` 349, `.jpg` 348, `.png` 348, `.webp` 55
- Preflight verified by sha256: 1,100
- Preflight bytes read: 43,916,396
- OCR workers: 11
- Tesseract internal thread limit: 1
- OCR wall time from command: 17.22 seconds
- OCR processing wall time from report: 16.444 seconds
- OCR bytes submitted: 43,916,396
- Extracted text: 961
- Empty text: 139
- OCR errors: 0
- OCR text characters: 75,835
- OCR text files present: 961
- Images/thumbnails written: 0
- Warnings: `tesseract_stderr_present` 10
- Leak check: JSON and SQLite contain no passwords, email values, absolute input paths, or raw OCR text

19. Run full safe image OCR batch:

```bash
python3 scripts/mango_office_mail_archive.py attachment-stage6-plan \
  --parse-plan-db _external_handoffs/mail_archive_2026-05-12/regru_edu/full_all_mail_combined_20260513/attachment_parse_plan_stage1/mail_attachment_parse_plan.sqlite \
  --text-index-db _external_handoffs/mail_archive_2026-05-12/regru_edu/full_all_mail_combined_20260513/attachment_text_index_stage5/mail_attachment_text_index.sqlite \
  --out-dir _external_handoffs/mail_archive_2026-05-12/regru_edu/full_all_mail_combined_20260513/attachment_ocr_full_safe_stage6_plan \
  --ocr-pilot-limit 29782 \
  --max-pilot-attachment-bytes 5000000

python3 scripts/mango_office_mail_archive.py attachment-ocr-preflight \
  --archive-db <verified_archive_1>/mail_archive.sqlite \
  --archive-db <verified_archive_2>/mail_archive.sqlite \
  --stage6-plan-db _external_handoffs/mail_archive_2026-05-12/regru_edu/full_all_mail_combined_20260513/attachment_ocr_full_safe_stage6_plan/mail_attachment_stage6_plan.sqlite \
  --out-dir _external_handoffs/mail_archive_2026-05-12/regru_edu/full_all_mail_combined_20260513/attachment_ocr_full_safe_preflight \
  --max-candidates 29782 \
  --max-attachment-bytes 5000000

python3 scripts/mango_office_mail_archive.py attachment-ocr-pilot \
  --archive-db <verified_archive_1>/mail_archive.sqlite \
  --archive-db <verified_archive_2>/mail_archive.sqlite \
  --ocr-preflight-db _external_handoffs/mail_archive_2026-05-12/regru_edu/full_all_mail_combined_20260513/attachment_ocr_full_safe_preflight/mail_attachment_ocr_preflight.sqlite \
  --out-dir _external_handoffs/mail_archive_2026-05-12/regru_edu/full_all_mail_combined_20260513/attachment_ocr_full_safe \
  --max-candidates 29782 \
  --max-attachment-bytes 5000000 \
  --languages rus+eng \
  --psm 6 \
  --tesseract-timeout-seconds 30 \
  --workers 11 \
  --tesseract-thread-limit 1 \
  --reuse-existing-ocr-text
```

Full safe OCR batch notes:

- This pass covers all OCR-supported image attachments under the 5 MB safety cap.
- Larger image attachments remain excluded from this pass and require a separate decision.
- The first full run found a real timeout-handling bug: `subprocess.TimeoutExpired` was escaping the per-file OCR handler and stopping the whole batch. The handler now converts it to `ocr_error/tesseract_timeout` without exposing the source path in reports.
- `--reuse-existing-ocr-text` is a recovery/resume mode. It still re-checks source attachment sha256 first, then reads existing derived OCR text under ignored `_external_handoffs` and writes only text hashes/counts into SQLite.

Full safe OCR output, built on 2026-05-14:

- OCR-supported candidates: 30,074
- Selected under 5 MB safety cap: 29,782
- Excluded over 5 MB: 292
- Selected by extension: `.jpeg` 3,105, `.jpg` 13,430, `.png` 13,192, `.webp` 55
- Preflight verified by sha256: 29,782
- Preflight bytes read: 11,792,811,373
- OCR workers: 11
- Tesseract internal thread limit: 1
- First full OCR attempt wall time before timeout bug: 1,438.17 seconds
- Resume pass wall time: 40.44 seconds
- Resume OCR processing wall time from report: 39.076 seconds
- OCR bytes submitted during resume: 35,004,559
- Reused existing OCR text files: 29,553
- Extracted text: 29,553
- Empty text: 214
- OCR errors: 15
- OCR error reasons: `tesseract_failed` 11, `tesseract_timeout` 4
- OCR text characters: 6,879,165
- OCR text files present: 29,553
- Images/thumbnails written: 0
- Leak check: JSON and SQLite contain no passwords, email values, absolute input paths, or raw OCR text

20. Run large image OCR tail:

```bash
python3 scripts/mango_office_mail_archive.py attachment-stage6-plan \
  --parse-plan-db _external_handoffs/mail_archive_2026-05-12/regru_edu/full_all_mail_combined_20260513/attachment_parse_plan_stage1/mail_attachment_parse_plan.sqlite \
  --text-index-db _external_handoffs/mail_archive_2026-05-12/regru_edu/full_all_mail_combined_20260513/attachment_text_index_stage5/mail_attachment_text_index.sqlite \
  --out-dir _external_handoffs/mail_archive_2026-05-12/regru_edu/full_all_mail_combined_20260513/attachment_ocr_large_100mb_stage6_plan \
  --ocr-pilot-limit 292 \
  --min-pilot-attachment-bytes 5000000 \
  --max-pilot-attachment-bytes 100000000

python3 scripts/mango_office_mail_archive.py attachment-ocr-preflight \
  --archive-db <verified_archive_1>/mail_archive.sqlite \
  --archive-db <verified_archive_2>/mail_archive.sqlite \
  --stage6-plan-db _external_handoffs/mail_archive_2026-05-12/regru_edu/full_all_mail_combined_20260513/attachment_ocr_large_100mb_stage6_plan/mail_attachment_stage6_plan.sqlite \
  --out-dir _external_handoffs/mail_archive_2026-05-12/regru_edu/full_all_mail_combined_20260513/attachment_ocr_large_100mb_preflight \
  --max-candidates 292 \
  --max-attachment-bytes 100000000

python3 scripts/mango_office_mail_archive.py attachment-ocr-pilot \
  --archive-db <verified_archive_1>/mail_archive.sqlite \
  --archive-db <verified_archive_2>/mail_archive.sqlite \
  --ocr-preflight-db _external_handoffs/mail_archive_2026-05-12/regru_edu/full_all_mail_combined_20260513/attachment_ocr_large_100mb_preflight/mail_attachment_ocr_preflight.sqlite \
  --out-dir _external_handoffs/mail_archive_2026-05-12/regru_edu/full_all_mail_combined_20260513/attachment_ocr_large_100mb \
  --max-candidates 292 \
  --max-attachment-bytes 100000000 \
  --languages rus+eng \
  --psm 6 \
  --tesseract-timeout-seconds 120 \
  --workers 6 \
  --tesseract-thread-limit 1
```

Large image OCR tail output, built on 2026-05-14:

- Selected over 5 MB and under 100 MB: 292
- Selected by extension: `.jpeg` 24, `.jpg` 236, `.png` 32
- Size range: 5,013,087 to 13,891,164 bytes
- Selected bytes: 1,912,246,019
- Preflight verified by sha256: 292
- Preflight wall time: 2.26 seconds
- OCR workers: 6
- Tesseract internal thread limit: 1
- Tesseract timeout: 120 seconds
- OCR wall time: 356.80 seconds
- OCR processing wall time from report: 356.348 seconds
- OCR bytes submitted: 1,912,246,019
- Extracted text: 291
- Empty text: 0
- OCR errors: 1
- OCR error reasons: `tesseract_failed` 1
- OCR text characters: 720,653
- OCR text files present: 291
- Images/thumbnails written: 0
- Leak check: JSON and SQLite contain no passwords, email values, absolute input paths, or raw OCR text

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
- attachment parse-plan output path is outside ignored `_external_handoffs`
- attachment parse-plan reads attachment bytes, raw `.eml`, or extracted text
- attachment parse-plan writes raw attachment filenames or raw attachment paths
- attachment text-extract output path is outside ignored `_external_handoffs`
- attachment text-extract reads anything except stage-supported `parse_later` attachments
- attachment text-extract writes raw filenames, raw source attachment paths, or extracted text into JSON reports
- attachment text-extract follows Office external links or parses macro/embedded-object Office files
- attachment PDF-extract output path is outside ignored `_external_handoffs`
- attachment PDF-extract reads anything except `.pdf` `parse_later` attachments
- attachment PDF-extract renders pages, extracts images, runs OCR, follows links, or allows PDF actions
- attachment PDF-extract writes raw filenames, raw source attachment paths, or extracted text into JSON reports
- attachment image OCR-plan output path is outside ignored `_external_handoffs`
- attachment image OCR-plan runs OCR, decodes images, extracts EXIF, writes thumbnails, or writes image files
- attachment image OCR-plan reads attachment bytes without an explicit small-pilot `--inspect-headers` run
- attachment image OCR-plan writes raw filenames, raw source attachment paths, raw image content, EXIF, or OCR text into JSON reports
- attachment text-index output path is outside ignored `_external_handoffs`
- attachment text-index reads derived text content, raw `.eml`, raw attachments, PDFs, images, or OCR output
- attachment text-index writes raw filenames, raw source attachment paths, raw derived text paths, extracted text, OCR text, email addresses, or phones into JSON reports
- attachment text-index treats `ocr_pending`, `blocked`, `empty_text`, `parse_error`, or `skipped` rows as available text
- attachment Stage 6 output path is outside ignored `_external_handoffs`
- attachment Stage 6 reads derived text content, raw `.eml`, raw attachments, PDFs, images, or OCR output
- attachment Stage 6 runs OCR, decodes images, extracts EXIF, writes thumbnails, or writes image files
- attachment Stage 6 writes raw filenames, raw source attachment paths, raw derived text paths, extracted text, OCR text, email addresses, or phones into JSON reports
- attachment Stage 6 selects more than the explicit `--ocr-pilot-limit` rows
- attachment OCR preflight output path is outside ignored `_external_handoffs`
- attachment OCR preflight reads any attachment outside selected Stage 6 OCR pilot candidates
- attachment OCR preflight decodes images, extracts EXIF, writes thumbnails, writes image files, runs OCR, or writes OCR text
- attachment OCR preflight writes raw filenames, raw source attachment paths, raw image content, EXIF, OCR text, email addresses, or phones into JSON reports
- attachment OCR preflight has any selected candidate that is not sha256-verified
- attachment OCR pilot output path is outside ignored `_external_handoffs`
- attachment OCR pilot reads any attachment outside verified preflight candidates
- attachment OCR pilot runs OCR before sha256 verification or over a file with mismatched sha256
- attachment OCR pilot writes images, thumbnails, EXIF, raw source paths, raw filenames, raw OCR text, email addresses, or phones into JSON/SQLite reports
- attachment OCR pilot writes derived OCR text outside ignored `_external_handoffs`
- attachment OCR pilot enables CRM/Tallanto writes, ASR, R+A, PDF rendering, archive extraction, external links, or `stable_runtime` writes
- attachment OCR pilot uses high `--workers` without `--tesseract-thread-limit 1`
- attachment OCR pilot reuses existing OCR text without first re-checking source attachment sha256
- attachment OCR pilot reads existing OCR text from outside the current ignored OCR output directory

## Commit-Safe Files

Mail archive code/docs/tests only:

```text
scripts/mango_office_mail_archive.py
src/mango_mvp/productization/mail_archive.py
tests/test_productization_mail_archive.py
docs/MAIL_ARCHIVE_RUNBOOK_2026-05-12.md
pyproject.toml
requirements.txt
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
