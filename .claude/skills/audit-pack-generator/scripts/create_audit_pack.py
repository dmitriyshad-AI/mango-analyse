#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from datetime import datetime
from pathlib import Path


FILES = {
    "implementation_notes.md": "# Implementation Notes\n\nBlock:\n\nCommits:\n\nWhat changed:\n\n",
    "changed_files.txt": "",
    "test_output.txt": "# Test Output\n\nCommands:\n\nResults:\n\n",
    "semantic_review.md": "# Semantic Review\n\nVerdict: `PENDING`\n\nAudience:\n\nWhat passed:\n\nRisks:\n\nMissing checks:\n\n",
    "risk_review.md": "# Risk Review\n\nHigh-impact risks:\n\nNegative controls:\n\nResidual risk:\n\n",
    "backward_compatibility.md": "# Backward Compatibility\n\nExpected unchanged behavior:\n\nNeighbor tests:\n\n",
}


def _git_changed_files(root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "HEAD"],
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
        )
    except Exception:
        return ""
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return "\n".join(lines) + ("\n" if lines else "")


def main() -> int:
    parser = argparse.ArgumentParser(description="Create Mango audit pack skeleton.")
    parser.add_argument("block_name", help="Short block name, e.g. a21_block1")
    parser.add_argument("--root", default=".", help="Repository root")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in args.block_name).strip("_")
    pack = root / "audits" / "_inbox" / f"{safe_name}_{stamp}"
    pack.mkdir(parents=True, exist_ok=False)

    for name, content in FILES.items():
        if name == "changed_files.txt":
            content = _git_changed_files(root)
        (pack / name).write_text(content, encoding="utf-8")

    print(pack)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
