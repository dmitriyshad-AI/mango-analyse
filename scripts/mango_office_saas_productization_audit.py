#!/usr/bin/env python3
"""Build a safe SaaS/productization audit over the disposable Mango DB.

The command is read-only for calls/audio/runtime data. It writes JSON reports
only under the selected productization output root.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mango_mvp.productization.insight_seed import build_insight_seed_report  # noqa: E402
from mango_mvp.productization.repository import ProductRepository  # noqa: E402
from mango_mvp.productization.supervisor import build_supervisor_dry_run_report  # noqa: E402
from mango_mvp.productization.tenant_owner_mapping import build_tenant_owner_mapping_draft  # noqa: E402
from mango_mvp.productization.test_ingest import path_is_relative_to  # noqa: E402
from mango_mvp.productization.ui_contracts import build_dashboard_contract  # noqa: E402


DEFAULT_ROOT = "_local_archive_mango_api_downloads_20260507/quarantine_import"
DEFAULT_DB = f"{DEFAULT_ROOT}/test_ingest/quarantine_test_ingest.sqlite"
DEFAULT_OUT_ROOT = f"{DEFAULT_ROOT}/test_ingest"
DEFAULT_RAW_PAYLOAD = f"{DEFAULT_ROOT}/raw_payload_archive/shadow_poll_raw_rows_20260506_20260507.jsonl"
DEFAULT_AUDIO_DIR = f"{DEFAULT_ROOT}/audio"
DEFAULT_OUT = f"{DEFAULT_OUT_ROOT}/saas_productization_pass_audit.json"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    out_root = Path(args.out_root)
    allowed_root = Path(args.allowed_root).resolve(strict=False)
    guard_output_paths(out_root.resolve(strict=False), Path(args.out).resolve(strict=False), allowed_root)
    repo = ProductRepository(db_path=Path(args.db), out_allowed_root=Path(args.allowed_root))

    owner_draft_path = out_root / "tenant_owner_mapping_draft.json"
    supervisor_path = out_root / "supervisor_dry_run_audit.json"
    ui_contract_path = out_root / "ui_dashboard_contract_sample.json"
    insight_path = out_root / "insight_seed_report.json"

    tenant_owner = build_tenant_owner_mapping_draft(
        db_path=Path(args.db),
        out_allowed_root=Path(args.allowed_root),
        out_path=owner_draft_path,
    )
    supervisor = build_supervisor_dry_run_report(
        repo=repo,
        raw_payload_paths=[Path(args.raw_payload)],
        quarantine_audio_dir=Path(args.audio_dir),
        out_path=supervisor_path,
    )
    ui_contract = build_dashboard_contract(repo=repo, call_limit=args.call_limit)
    write_json(ui_contract_path, ui_contract)
    insight = build_insight_seed_report(repo=repo, max_evidence_per_manager=args.max_evidence_per_manager)
    write_json(insight_path, insight)

    report = {
        "summary": {
            "validation_ok": bool(
                tenant_owner["summary"]["validation_ok"]
                and supervisor["summary"]["validation_ok"]
                and insight["summary"]["validation_ok"]
            ),
            "repository": repo.summary().to_json_dict(),
            "tenant_owner_mapping": tenant_owner["summary"],
            "supervisor": supervisor["summary"],
            "ui_contract": {
                "schema_version": ui_contract["schema_version"],
                "calls_sampled": len(ui_contract["views"]["call_list"]["items"]),
                "manager_filters": len(ui_contract["filters"]["managers"]),
                "manual_review_items": len(ui_contract["views"]["manual_owner_review_queue"]["items"]),
            },
            "insight_seed": insight["summary"],
        },
        "outputs": {
            "tenant_owner_mapping_draft": str(owner_draft_path),
            "supervisor_dry_run_audit": str(supervisor_path),
            "ui_dashboard_contract_sample": str(ui_contract_path),
            "insight_seed_report": str(insight_path),
        },
        "safety": {
            "runtime_db_writes": False,
            "stable_runtime_writes": False,
            "asr_run": False,
            "ra_run": False,
            "crm_writes": False,
        },
    }
    out_path = Path(args.out)
    write_json(out_path, report)
    print(json.dumps({"out": str(out_path), "summary": report["summary"]}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["summary"]["validation_ok"] else 1


def guard_output_paths(out_root: Path, out_path: Path, allowed_root: Path) -> None:
    if not path_is_relative_to(out_root, allowed_root):
        raise ValueError(f"out-root must stay under allowed root: {allowed_root}")
    if not path_is_relative_to(out_path, allowed_root):
        raise ValueError(f"out path must stay under allowed root: {allowed_root}")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SaaS/productization pass audit.")
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--allowed-root", default=DEFAULT_ROOT)
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--raw-payload", default=DEFAULT_RAW_PAYLOAD)
    parser.add_argument("--audio-dir", default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--call-limit", type=int, default=50)
    parser.add_argument("--max-evidence-per-manager", type=int, default=3)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
