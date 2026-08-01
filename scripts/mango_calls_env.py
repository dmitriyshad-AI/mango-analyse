#!/usr/bin/env python3
"""Parse the Mango calls worker env without executing shell syntax."""

from __future__ import annotations

import argparse
import re
import shlex
from pathlib import Path
from typing import Sequence


KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
FORBIDDEN_VALUE_CHARS = "$`;|&<>()"


def parse_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"line {line_number}: expected KEY=VALUE")
        key, raw_value = line.split("=", 1)
        key, raw_value = key.strip(), raw_value.strip()
        if not KEY_RE.fullmatch(key) or key in values:
            raise ValueError(f"line {line_number}: invalid or duplicate key")
        if any(char in raw_value for char in FORBIDDEN_VALUE_CHARS):
            raise ValueError(f"line {line_number}: shell syntax is forbidden")
        try:
            tokens = [""] if not raw_value else shlex.split(raw_value, comments=False, posix=True)
        except ValueError as exc:
            raise ValueError(f"line {line_number}: invalid quoting") from exc
        if len(tokens) != 1 or "\n" in tokens[0] or "\x00" in tokens[0]:
            raise ValueError(f"line {line_number}: value must be one literal token")
        values[key] = tokens[0]
    return values


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--get")
    parser.add_argument("--export-lines", action="store_true")
    args = parser.parse_args(argv)
    values = parse_env(args.path)
    if args.get is not None:
        value = values.get(args.get, "")
        if not value:
            return 1
        print(value)
    elif args.export_lines:
        for key, value in values.items():
            print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
