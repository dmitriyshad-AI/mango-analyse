# Customer Timeline daily capture launchd package

This package only prepares launchd templates. Installing or running them is a
manual owner action.

## Services

- `com.mango.customer-timeline-mango-capture`: prepares the daily Mango calls
  capture wrapper. It does not run ASR unless the wrapped command is explicitly
  configured by the owner through `MANGO_CAPTURE_COMMAND_FILE`.
- `com.mango.customer-timeline-mail-chain` at 02:00: runs
  `mail-download -> mail-process -> mail-import` sequentially through
  `scripts/run_customer_timeline_codex_task.py`. `mail-process` starts only
  after successful download; `mail-import` starts only after successful
  process. The chain stops on the first `stopped`/failed stage.
- The older split templates `com.mango.customer-timeline-mail-download`,
  `com.mango.customer-timeline-mail-process`, and
  `com.mango.customer-timeline-mail-import` are disabled compatibility
  artifacts and must not be installed as schedules.
- The older `com.mango.customer-timeline-mail-capture.plist.template` is
  **deprecated and must never be installed**. It is retained only as a tracked
  compatibility artifact until a separately approved deletion.
- `com.mango.customer-timeline-tallanto-api-capture`: runs the same Codex-task
  wrapper as launchd. It is fail-closed until `TALLANTO_API_CAPTURE_ENABLED=1`
  is set for an explicitly approved read-only Tallanto API capture.

## Dry-run checks

```bash
python3 scripts/run_customer_timeline_mail_chain.py --help
python3 scripts/run_customer_timeline_mail_download.py
bash scripts/run_customer_timeline_mango_capture_daily.sh
python3 scripts/run_customer_timeline_codex_task.py --help
bash scripts/install_customer_timeline_nightly_service.sh \
  --plist deploy/customer_timeline_daily_captures/com.mango.customer-timeline-mail-chain.plist.template \
  --target /tmp/com.mango.customer-timeline-mail-chain.plist
bash scripts/uninstall_customer_timeline_nightly_service.sh \
  --target /tmp/com.mango.customer-timeline-mail-chain.plist
```

## Manual install after owner approval

```bash
bash scripts/install_customer_timeline_nightly_service.sh \
  --plist deploy/customer_timeline_daily_captures/com.mango.customer-timeline-mail-chain.plist.template \
  --code-root "/absolute/permanent/main/worktree" \
  --target "$HOME/Library/LaunchAgents/com.mango.customer-timeline-mail-chain.plist" --apply

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
an audit pack, and a separate owner approval. The code worktree in the installed
template is a permanent runtime dependency while the agent is installed; do not
remove or switch it.
