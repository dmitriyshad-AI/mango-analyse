from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

from .pseudonymizer import BIRTH_DATE_CONTEXT_RE, BIRTH_DATE_RE, pii_findings


def iter_scan_files(paths: Iterable[Path]) -> Iterable[Path]:
    for raw_path in paths:
        path = raw_path.expanduser()
        if path.is_dir():
            for nested in sorted(item for item in path.rglob("*") if item.is_file()):
                yield nested
        elif path.is_file():
            yield path


def load_scan_payloads(path: Path) -> Iterable[tuple[str, Any]]:
    suffix = path.suffix.casefold()
    if suffix == ".sha256":
        yield str(path), {"sha256": path.read_text(encoding="utf-8", errors="replace").strip()}
        return
    if suffix == ".csv":
        with path.open(encoding="utf-8", errors="replace", newline="") as handle:
            for line_no, row in enumerate(csv.DictReader(handle), start=2):
                structured: dict[str, Any] = {}
                for key, value in row.items():
                    text = str(value or "").strip()
                    if text.startswith(("{", "[")):
                        try:
                            structured[str(key)] = json.loads(text)
                            continue
                        except json.JSONDecodeError:
                            pass
                    structured[str(key)] = value
                yield f"{path}:{line_no}", structured
        return
    text = path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".jsonl":
        for line_no, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                yield f"{path}:{line_no}", json.loads(line)
            except json.JSONDecodeError:
                yield f"{path}:{line_no}", line
        return
    if suffix == ".json":
        try:
            yield str(path), json.loads(text)
            return
        except json.JSONDecodeError:
            pass
    lines = text.splitlines()
    for line_no, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("{", "[")):
            try:
                yield f"{path}:{line_no}", json.loads(stripped)
                continue
            except json.JSONDecodeError:
                pass
        previous = lines[line_no - 2] if line_no > 1 else ""
        payload = f"{previous}\n{line}" if BIRTH_DATE_RE.search(line) and BIRTH_DATE_CONTEXT_RE.search(previous) else line
        yield f"{path}:{line_no}", payload


def scan_paths(paths: Iterable[Path], *, allowlist: Iterable[str] = ()) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    for path in iter_scan_files(paths):
        for source, payload in load_scan_payloads(path):
            for finding in pii_findings(payload, allowlist=allowlist):
                item = dict(finding)
                item["source"] = source
                findings.append(item)
    findings.sort(key=lambda item: (item.get("source", ""), item.get("path", ""), item.get("kind", "")))
    return findings
