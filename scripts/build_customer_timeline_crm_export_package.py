#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mango_mvp.customer_timeline.crm_export_package import (  # noqa: E402
    CrmExportPackageConfig,
    build_crm_export_package,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a staging-only CRM export package from Customer Timeline.")
    parser.add_argument("--timeline-db", required=True, type=Path)
    parser.add_argument("--allowed-root", default=ROOT, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--tenant-id", default="foton")
    parser.add_argument("--pilot-size", type=int, default=20)
    parser.add_argument("--batch-limit", type=int, default=0)
    parser.add_argument("--customer-id", action="append", default=[])
    parser.add_argument("--canonical-calls-db", type=Path)
    parser.add_argument("--verify-idempotent", action="store_true")
    args = parser.parse_args(argv)

    config = CrmExportPackageConfig(
        timeline_db_path=args.timeline_db,
        allowed_root=args.allowed_root,
        out_dir=args.out_dir,
        tenant_id=args.tenant_id,
        pilot_size=args.pilot_size,
        batch_limit=args.batch_limit,
        customer_ids=tuple(args.customer_id or ()),
        canonical_calls_db_path=args.canonical_calls_db,
    )
    summary = dict(build_crm_export_package(config))
    if args.verify_idempotent:
        repeat_dir = args.out_dir.with_name(args.out_dir.name + "_repeat")
        repeat_summary = dict(
            build_crm_export_package(
                CrmExportPackageConfig(
                    timeline_db_path=args.timeline_db,
                    allowed_root=args.allowed_root,
                    out_dir=repeat_dir,
                    tenant_id=args.tenant_id,
                    pilot_size=args.pilot_size,
                    batch_limit=args.batch_limit,
                    customer_ids=tuple(args.customer_id or ()),
                    canonical_calls_db_path=args.canonical_calls_db,
                )
            )
        )
        first_sha = summary.get("output_sha256") or {}
        repeat_sha = repeat_summary.get("output_sha256") or {}
        summary["idempotence"] = {
            "checked": True,
            "passed": first_sha == repeat_sha,
            "repeat_manifest": repeat_summary.get("manifest_path"),
        }
        if first_sha != repeat_sha:
            print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
            return 2
        manifest_path = Path(str(summary["manifest_path"]))
        manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_payload["idempotence"] = summary["idempotence"]
        manifest_path.write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
