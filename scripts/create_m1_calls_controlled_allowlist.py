#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mango_mvp.customer_timeline.calls_two_processes import (  # noqa: E402
    CallsTwoProcessesConfig,
    controlled_call_database_snapshot,
    controlled_read_only_cutover_authority_report,
    process_a_leases,
)
from mango_mvp.productization.mango_calls_service_contract import (  # noqa: E402
    current_git_sha,
    git_worktree_is_clean,
    read_host_id,
)
from mango_mvp.productization.owner_only_io import (  # noqa: E402
    atomic_replace_owner_only_bytes,
    path_has_cloud_marker,
    validate_owner_only_directory,
)
from mango_mvp.services.controlled_call_scope import (  # noqa: E402
    CONTROLLED_CALL_ALLOWLIST_SCHEMA,
    load_controlled_call_allowlist,
)


def canonical_source_call_id(value: str) -> str:
    if value != value.strip() or not value or len(value) > 256:
        raise RuntimeError("source_call_id must be one canonical non-empty string")
    if any(ord(char) < 32 for char in value):
        raise RuntimeError("source_call_id contains a control character")
    return value


def verify_controlled_cutover_lineage(
    config: CallsTwoProcessesConfig,
) -> None:
    """Verify the transferred cursor without enabling service cutover."""
    authority = controlled_read_only_cutover_authority_report(config)
    if not (
        authority.get("ok") is True
        and authority.get("controlled_cursor_binding_ok") is True
    ):
        raise RuntimeError("controlled_allowlist_cutover_lineage_unproven")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create one owner-only Mango Calls controlled-one allowlist. "
            "Does not call Mango, models, or external systems."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--source-call-id", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    previous_umask = os.umask(0o077)
    try:
        config = CallsTwoProcessesConfig.from_json(args.config)
        if config.processing_scope != "service":
            raise RuntimeError(
                "create the allowlist from the clean service config, then make a separate controlled-one config"
            )
        source_call_id = canonical_source_call_id(args.source_call_id)
        if not re.fullmatch(r"[0-9a-f]{40}", config.expected_code_sha or ""):
            raise RuntimeError("expected_code_sha is missing or invalid")
        actual_sha = current_git_sha(ROOT)
        if actual_sha != config.expected_code_sha or not git_worktree_is_clean(ROOT):
            raise RuntimeError("Git worktree must be clean at expected_code_sha")
        host_id = read_host_id(config.host_id_file)
        if host_id != config.expected_active_host_id:
            raise RuntimeError("active host_id does not match config")

        with process_a_leases(
            config,
            pipeline_lock_info=None,
            skip_capture=False,
        ):
            # Freeze the transferred cursor and target under both local locks.
            db_snapshot = controlled_call_database_snapshot(
                config.working_db,
                source_call_id,
                working_audio_dir=config.working_audio_dir,
                require_source_audio=True,
            )
            verify_controlled_cutover_lineage(config)
            out = args.out.expanduser()
            if not out.is_absolute():
                raise RuntimeError("allowlist output path must be absolute")
            owner_local = (Path.home() / ".mango_local").resolve(strict=True)
            resolved_parent = out.parent.resolve(strict=True)
            if (
                owner_local not in resolved_parent.parents
                and resolved_parent != owner_local
            ):
                raise RuntimeError("allowlist output must stay below $HOME/.mango_local")
            if path_has_cloud_marker(resolved_parent):
                raise RuntimeError("allowlist output must stay outside cloud folders")
            validate_owner_only_directory(
                resolved_parent,
                label="controlled_call_allowlist_parent",
                owner_only_mode=0o700,
            )
            payload = {
                "schema_version": CONTROLLED_CALL_ALLOWLIST_SCHEMA,
                "source_call_ids": [source_call_id],
                "target_record_id": db_snapshot["target"]["record_id"],
                "source_audio_sha256": db_snapshot["target"]["source_audio"][
                    "sha256"
                ],
                "source_audio_size_bytes": db_snapshot["target"]["source_audio"][
                    "size_bytes"
                ],
                "tenant_id": config.tenant_id,
                "code_sha": actual_sha,
                "host_id": host_id,
            }
            raw = json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            atomic_replace_owner_only_bytes(
                out,
                raw,
                label="controlled_call_allowlist",
            )
            digest = hashlib.sha256(raw).hexdigest()
            loaded = load_controlled_call_allowlist(
                path=out,
                expected_sha256=digest,
                expected_tenant_id=config.tenant_id,
                expected_code_sha=actual_sha,
                expected_host_id=host_id,
                host_id_path=config.host_id_file,
                project_root=ROOT,
            )
        print(
            json.dumps(
                {
                    "status": "ok",
                    "path": str(loaded.allowlist_path),
                    "sha256": loaded.allowlist_sha256,
                    "source_call_id_sha256": hashlib.sha256(
                        source_call_id.encode("utf-8")
                    ).hexdigest(),
                    "database_target_row_sha256": db_snapshot[
                        "target_row_sha256"
                    ],
                    "runs_models": False,
                    "writes_external_systems": False,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except Exception as exc:  # noqa: BLE001 - fail closed with no identifier echo.
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error": type(exc).__name__,
                    "runs_models": False,
                    "writes_external_systems": False,
                },
                sort_keys=True,
            )
        )
        return 1
    finally:
        os.umask(previous_umask)


if __name__ == "__main__":
    raise SystemExit(main())
