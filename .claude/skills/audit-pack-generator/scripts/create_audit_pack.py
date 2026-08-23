#!/usr/bin/env python3
"""Compatibility wrapper for the canonical Mango audit-pack builder."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[4]
    raise SystemExit(subprocess.run([sys.executable, str(root / "scripts/make_audit_pack.py"), *sys.argv[1:]], cwd=root).returncode)
