# Customer Timeline daily capture launchd package

This package only prepares launchd templates. Installing or running them is a
manual owner action.

## Services

- `com.mango.customer-timeline-mango-capture`: prepares the daily Mango calls
  capture wrapper. It does not run ASR unless the wrapped command is explicitly
  configured by the owner through `MANGO_CAPTURE_COMMAND_FILE`.
- `com.mango.customer-timeline-mail-capture`: rebuilds staging-local mail
  handoff inputs for the nightly service from existing local mail archives.
- `com.mango.customer-timeline-tallanto-api-capture`: runs the same Codex-task
  wrapper as launchd. It is fail-closed until `TALLANTO_API_CAPTURE_ENABLED=1`
  is set for an explicitly approved read-only Tallanto API capture.

## Dry-run checks

```bash
bash scripts/run_customer_timeline_mail_capture_daily.sh
bash scripts/run_customer_timeline_mango_capture_daily.sh
python3 scripts/run_customer_timeline_codex_task.py --task tallanto-api-capture
bash scripts/install_customer_timeline_nightly_service.sh \
  --plist deploy/customer_timeline_daily_captures/com.mango.customer-timeline-mail-capture.plist.template \
  --target /tmp/com.mango.customer-timeline-mail-capture.plist
bash scripts/uninstall_customer_timeline_nightly_service.sh \
  --target /tmp/com.mango.customer-timeline-mail-capture.plist
```

## Manual install after owner approval

```bash
bash scripts/install_customer_timeline_nightly_service.sh \
  --plist deploy/customer_timeline_daily_captures/com.mango.customer-timeline-mail-capture.plist.template \
  --target "$HOME/Library/LaunchAgents/com.mango.customer-timeline-mail-capture.plist" \
  --apply

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
