#!/usr/bin/env python3
"""Read-only customer timeline import CLI.

Default mode is a dry-run preview. Use --apply only when you intentionally want
to write the isolated product timeline DB. This script does not call CRM,
Tallanto, messengers, ASR, R+A, or stable_runtime DBs.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mango_mvp.customer_timeline.import_cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
