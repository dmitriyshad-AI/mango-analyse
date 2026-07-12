# Customer Timeline daily capture launchd package

This package only prepares launchd templates. Installing or running them is a
manual owner action.

## Services

- `com.mango.customer-timeline-mango-capture`: prepares the daily Mango calls
  capture wrapper. It does not run ASR unless the wrapped command is explicitly
  configured by the owner through `MANGO_CAPTURE_COMMAND_FILE`.
- `com.mango.customer-timeline-mail-download` at 02:00: downloads `INBOX` and
  `Sent` through read-only IMAP into the canonical local archive.
- `com.mango.customer-timeline-mail-process` at 02:30: builds a mail-only
  increment after checking the fresh download manifest and staging cursor.
- `com.mango.customer-timeline-mail-import` at 02:50: imports that increment
  into the configured staging timeline after checking the process manifest.
- The older `com.mango.customer-timeline-mail-capture.plist.template` is
  **deprecated and must never be installed**. It is retained only as a tracked
  compatibility artifact until a separately approved deletion.
- `com.mango.customer-timeline-tallanto-api-capture`: runs the same Codex-task
  wrapper as launchd. It is fail-closed until `TALLANTO_API_CAPTURE_ENABLED=1`
  is set for an explicitly approved read-only Tallanto API capture.

## Dry-run checks

```bash
python3 scripts/run_customer_timeline_mail_download.py
python3 scripts/run_customer_timeline_codex_task.py --task mail-process
python3 scripts/run_customer_timeline_codex_task.py --task mail-import
bash scripts/run_customer_timeline_mango_capture_daily.sh
python3 scripts/run_customer_timeline_codex_task.py --task tallanto-api-capture
bash scripts/install_customer_timeline_nightly_service.sh \
  --plist deploy/customer_timeline_daily_captures/com.mango.customer-timeline-mail-download.plist.template \
  --target /tmp/com.mango.customer-timeline-mail-download.plist
bash scripts/uninstall_customer_timeline_nightly_service.sh \
  --target /tmp/com.mango.customer-timeline-mail-download.plist
```

## Manual install after owner approval

```bash
bash scripts/install_customer_timeline_nightly_service.sh \
  --plist deploy/customer_timeline_daily_captures/com.mango.customer-timeline-mail-download.plist.template \
  --target "$HOME/Library/LaunchAgents/com.mango.customer-timeline-mail-download.plist" --apply

bash scripts/install_customer_timeline_nightly_service.sh \
  --plist deploy/customer_timeline_daily_captures/com.mango.customer-timeline-mail-process.plist.template \
  --target "$HOME/Library/LaunchAgents/com.mango.customer-timeline-mail-process.plist" --apply

bash scripts/install_customer_timeline_nightly_service.sh \
  --plist deploy/customer_timeline_daily_captures/com.mango.customer-timeline-mail-import.plist.template \
  --target "$HOME/Library/LaunchAgents/com.mango.customer-timeline-mail-import.plist" --apply

bash scripts/install_customer_timeline_nightly_service.sh \
  --plist deploy/customer_timeline_daily_captures/com.mango.customer-timeline-mango-capture.plist.template \
  --target "$HOME/Library/LaunchAgents/com.mango.customer-timeline-mango-capture.plist" \
  --apply

bash scripts/install_customer_timeline_nightly_service.sh \
  --plist deploy/customer_timeline_daily_captures/com.mango.customer-timeline-tallanto-api-capture.plist.template \
  --target "$HOME/Library/LaunchAgents/com.mango.customer-timeline-tallanto-api-capture.plist" \
  --apply
```

Uninstall uses the same generic dry-run-safe uninstaller with the matching
`--target` and `--apply`.

Do not install or bootstrap these templates before three clean manual cycles,
an audit pack, and a separate owner approval. The code worktree in all three
templates is a permanent runtime dependency while the agents are installed;
do not remove or switch it.
