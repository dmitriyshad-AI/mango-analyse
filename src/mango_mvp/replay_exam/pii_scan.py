from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .pseudonymizer import pii_findings


def iter_scan_files(paths: Iterable[Path]) -> Iterable[Path]:
    for raw_path in paths:
        path = raw_path.expanduser()
        if path.is_dir():
            for nested in sorted(item for item in path.rglob("*") if item.is_file()):
                yield nested
        elif path.is_file():
            yield path


def load_scan_payloads(path: Path) -> Iterable[tuple[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix == ".jsonl":
        for line_no, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                yield f"{path}:{line_no}", json.loads(line)
            except json.JSONDecodeError:
                yield f"{path}:{line_no}", line
        return
    if path.suffix == ".json":
        try:
            yield str(path), json.loads(text)
            return
        except json.JSONDecodeError:
            pass
    yield str(path), text


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
