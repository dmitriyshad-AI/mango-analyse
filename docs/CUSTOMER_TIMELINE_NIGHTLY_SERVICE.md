# Customer Timeline Nightly Service

This is a staging-only service package for Marathon 2 Block 5.

## What it does

- Runs configured local incremental source steps through `run_nightly_incremental`.
- Writes only to the configured staging `customer_timeline.sqlite`.
- Publishes a snapshot manifest with SQLite/WAL/SHM hashes, table counts, source counts, and ingestion cursors.
- Uses a service-level lock plus the existing per-nightly lock.

## What it does not do

- Does not write prod DB.
- Does not write AMO/CRM/Tallanto.
- Does not send messages.
- Does not install launchd by itself.
- Does not run arbitrary shell commands from config.

## Install package

Template:

`deploy/customer_timeline_nightly/com.mango.customer-timeline-nightly.plist.template`

Dry-run install check:

```bash
bash scripts/install_customer_timeline_nightly_service.sh \
  --plist deploy/customer_timeline_nightly/com.mango.customer-timeline-nightly.plist.template
```

Actual install is intentionally gated:

```bash
bash scripts/install_customer_timeline_nightly_service.sh \
  --plist deploy/customer_timeline_nightly/com.mango.customer-timeline-nightly.plist.template \
  --apply
```

Run only after owner approval and after choosing the host.

Dry-run uninstall check:

```bash
bash scripts/uninstall_customer_timeline_nightly_service.sh
```

Actual uninstall:

```bash
bash scripts/uninstall_customer_timeline_nightly_service.sh --apply
```

## Staging config

The real staging config is local and ignored:

`.codex_local/staging/nightly_service/customer_timeline_nightly_service_config.json`

It should keep all paths under `.codex_local/staging`.

