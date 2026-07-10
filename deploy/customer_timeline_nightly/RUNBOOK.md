# Customer Timeline nightly launchd runbook

1. Validate the plist: `plutil -lint deploy/customer_timeline_nightly/com.mango.customer-timeline-nightly.plist.template`.
2. Dry-run install: `bash scripts/install_customer_timeline_nightly_service.sh --plist deploy/customer_timeline_nightly/com.mango.customer-timeline-nightly.plist.template`.
3. Install only after owner approval: add the same command with `--apply`.
4. Check status: `launchctl print gui/$(id -u)/com.mango.customer-timeline-nightly`.
5. Logs: `.codex_local/staging/nightly_service/launchd.stdout.log` and `launchd.stderr.log`.
6. The service runs at 03:30 and writes only under `.codex_local/staging`.
7. If the DB lock is busy, the service waits up to `lock_timeout_seconds`, then reports partial/failed instead of forcing.
8. The latest snapshot is updated only when required steps pass.
9. Uninstall: `bash scripts/uninstall_customer_timeline_nightly_service.sh --apply`.
10. SWAP/live enablement is not part of this service and stays manual.

