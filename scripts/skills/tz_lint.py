#!/usr/bin/env python3
"""Lint a Mango TZ before implementation."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from scripts.preflight import parse_tz_header


FOREIGN_USER_PATH_RE = re.compile(r"/Users/(dmitriy|dmitry|dima)(?:/|$)", re.I)
SHA_RE = re.compile(r"\b[0-9a-f]{7,40}\b", re.I)
OLD_VERSION_RE = re.compile(r"\b(old|legacy|стар(?:ый|ые|ого|ая)|замен[её]н|устаревш)\b", re.I)
BARE_LINE_REF_RE = re.compile(r"\b[\w./-]+\.py:\d{2,5}\b")


@dataclass(frozen=True)
class TzLintIssue:
    code: str
    line: int
    message: str


@dataclass(frozen=True)
class TzLintResult:
    status: str
    path: str
    header: dict[str, Any]
    issues: list[TzLintIssue]


def _line_no(text: str, index: int) -> int:
    return text.count("\n", 0, index) + 1


def lint_tz(path: Path) -> TzLintResult:
    text = path.read_text(encoding="utf-8", errors="ignore")
    header = parse_tz_header(text)
    issues: list[TzLintIssue] = []

    if not header.branch:
        issues.append(TzLintIssue("missing_header_branch", 1, "В машинной шапке нет поля 'Ветка'."))
    if not header.zones:
        issues.append(TzLintIssue("missing_header_zones", 1, "В машинной шапке нет поля 'Зоны'."))
    if not header.test_cmd:
        issues.append(TzLintIssue("missing_header_test_command", 1, "В машинной шапке нет поля 'Тест-команда'."))
    if not header.semantic:
        issues.append(TzLintIssue("missing_header_semantic", 1, "В машинной шапке нет поля 'Семантический-аудит'."))

    for lineno, line in enumerate(text.splitlines(), start=1):
        if FOREIGN_USER_PATH_RE.search(line) and "/Users/dmitrijfabarisov/" not in line:
            issues.append(
                TzLintIssue(
                    "foreign_user_path",
                    lineno,
                    "Найден путь другой машины; укажи переносимый путь или явно опиши оба варианта.",
                )
            )
        if SHA_RE.search(line) and OLD_VERSION_RE.search(line):
            issues.append(
                TzLintIssue(
                    "old_sha_tail",
                    lineno,
                    "Похоже на хвост старой версии или старый sha; привяжи к актуальному source-of-truth.",
                )
            )
        if BARE_LINE_REF_RE.search(line) and not re.search(r"\b(def|class|функц|символ|якор|по имени)\b", line, re.I):
            issues.append(
                TzLintIssue(
                    "bare_line_number",
                    lineno,
                    "Есть ссылка file.py:line без символа/якоря; при сдвиге строк ТЗ станет хрупким.",
                )
            )

    if not re.search(r"(^|\n)#{1,4}\s*(?:S\d+\.\s*)?При[её]мка\b|(^|\n).*При[её]мка\s*:", text, re.I):
        issues.append(TzLintIssue("missing_acceptance", 1, "Нет явного раздела 'Приёмка'."))
    if not re.search(r"(^|\n)#{1,4}\s*(?:S\d+\.\s*)?СТОП\b|(^|\n).*СТОП\s*:", text, re.I):
        issues.append(TzLintIssue("missing_stop", 1, "Нет явного раздела 'СТОП'."))

    return TzLintResult(
        status="PASS" if not issues else "FAIL",
        path=str(path),
        header=asdict(header),
        issues=issues,
    )


def _print_text(result: TzLintResult) -> None:
    print(f"{result.status}: {result.path}")
    if result.issues:
        for issue in result.issues:
            print(f"- {issue.code}:{issue.line}: {issue.message}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Lint a TZ before taking it into work.")
    parser.add_argument("tz", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    result = lint_tz(args.tz.expanduser())
    if args.json:
        print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    else:
        _print_text(result)
    return 0 if result.status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
