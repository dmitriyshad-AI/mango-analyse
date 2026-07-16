# Customer Timeline nightly launchd runbook

1. Validate the plist: `plutil -lint deploy/customer_timeline_nightly/com.mango.customer-timeline-nightly.plist.template`.
2. Dry-run install: `bash scripts/install_customer_timeline_nightly_service.sh --plist deploy/customer_timeline_nightly/com.mango.customer-timeline-nightly.plist.template --code-root "$PWD" --nightly-home "$HOME/.mango_local/customer_timeline_nightly"`.
3. Install only after owner approval: add the same command with `--apply`.
4. Check status: `launchctl print gui/$(id -u)/com.mango.customer-timeline-nightly`.
5. Logs: `$CUSTOMER_TIMELINE_NIGHTLY_HOME/.codex_local/staging/nightly_service/launchd.stdout.log` and `launchd.stderr.log`.
6. The service runs at 03:30 and writes only under `$CUSTOMER_TIMELINE_NIGHTLY_HOME/.codex_local/staging`; this persistent state is outside the code worktree.
7. If the DB lock is busy, the service waits up to `lock_timeout_seconds`, then reports partial/failed instead of forcing.
8. The latest snapshot is updated only when required steps pass.
9. Uninstall: `bash scripts/uninstall_customer_timeline_nightly_service.sh --apply`.
10. SWAP/live enablement is not part of this service and stays manual.
11. Missing staging DB or missing/unreadable `mango_calls_ready.sqlite` is a hard stop. The optional base config only adds already prepared AMO sources; without it self-heal deliberately builds the minimum independent chain: processed Mango calls plus mail. The result must still contain required calls, Mango sweep with the explicit ready DB, and mail steps.
