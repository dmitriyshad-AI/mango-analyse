#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mango_mvp.replay_exam.pii_scan import scan_paths
from mango_mvp.replay_exam.pseudonymizer import kb_contact_allowlist


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan scrubbed Wappi replay artifacts for PII leaks.")
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--snapshot", type=Path, help="Knowledge snapshot; official public contacts are allowlisted.")
    parser.add_argument("--out", type=Path, help="Write scanner report JSON.")
    args = parser.parse_args()

    allowlist: tuple[str, ...] = ()
    if args.snapshot:
        allowlist = kb_contact_allowlist(args.snapshot)
    findings = scan_paths(args.paths, allowlist=allowlist)
    report = {"schema_version": "wappi_replay_pii_scan_v2", "leak_count": len(findings), "findings": findings}
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.expanduser().parent.mkdir(parents=True, exist_ok=True)
        args.out.expanduser().write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
