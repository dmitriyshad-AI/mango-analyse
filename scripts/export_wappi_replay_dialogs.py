#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mango_mvp.replay_exam.exporter import assert_raw_output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay raw exporter guard. Live Wappi read requires explicit flag.")
    parser.add_argument("--raw-out", required=True, type=Path)
    parser.add_argument("--allow-live-wappi-read", action="store_true")
    args = parser.parse_args()
    out = assert_raw_output_path(args.raw_out)
    if not args.allow_live_wappi_read:
        raise SystemExit("Refusing live Wappi read without --allow-live-wappi-read and owner confirmation.")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"schema_version": "wappi_replay_raw_v1", "messages": []}, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"raw_export={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
