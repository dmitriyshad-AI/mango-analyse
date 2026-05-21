from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from mango_mvp.knowledge_base.answer_registry import (
    load_answer_registry,
    semantic_passed,
    validate_answer_registry,
    validate_draft_semantics,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run semantic gates for KB answer registry and MEGA draft outputs.")
    parser.add_argument("--bot-pack-dir", type=Path, required=True)
    parser.add_argument("--mega-results-jsonl", type=Path, default=None)
    parser.add_argument("--fixtures", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    source_ids = _load_source_ids(args.bot_pack_dir / "source_registry.json")
    answer_registry_path = _find_answer_registry(args.bot_pack_dir)
    registry_entries = load_answer_registry(answer_registry_path) if answer_registry_path else []
    registry_issues = validate_answer_registry(registry_entries, known_source_ids=source_ids)

    fixtures = _read_jsonl_by_id(args.fixtures) if args.fixtures else {}
    draft_issues = []
    draft_rows = []
    if args.mega_results_jsonl:
        for row in _read_jsonl(args.mega_results_jsonl):
            fixture = fixtures.get(str(row.get("test_id") or ""), {})
            allowed_markers = fixture.get("expected_in_draft") or []
            issues = validate_draft_semantics(
                draft_text=str(row.get("draft_text") or ""),
                brand=str(row.get("brand") or ""),
                route=str(row.get("actual_route") or ""),
                priority=str(row.get("priority") or ""),
                category=str(row.get("category") or ""),
                subcategory=str(row.get("subcategory") or ""),
                test_id=str(row.get("test_id") or ""),
                allowed_numeric_markers=[str(item) for item in allowed_markers],
            )
            draft_issues.extend(issues)
            for issue in issues:
                draft_rows.append(
                    {
                        "test_id": issue.test_id,
                        "code": issue.code,
                        "severity": issue.severity,
                        "message": issue.message,
                        "brand": str(row.get("brand") or ""),
                        "priority": str(row.get("priority") or ""),
                        "category": str(row.get("category") or ""),
                        "subcategory": str(row.get("subcategory") or ""),
                        "draft_text": str(row.get("draft_text") or ""),
                    }
                )

    all_issues = list(registry_issues) + list(draft_issues)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "bot_pack_dir": str(args.bot_pack_dir),
        "answer_registry_path": str(answer_registry_path) if answer_registry_path else "",
        "answer_registry_entries": len(registry_entries),
        "registry_issues": [issue.to_json_dict() for issue in registry_issues],
        "draft_issues": [issue.to_json_dict() for issue in draft_issues],
        "summary": {
            "formal_pass": True,
            "semantic_pass": semantic_passed(all_issues),
            "registry_issue_count": len(registry_issues),
            "draft_issue_count": len(draft_issues),
            "errors": sum(1 for issue in all_issues if issue.severity == "error"),
            "warnings": sum(1 for issue in all_issues if issue.severity == "warning"),
            "by_code": dict(Counter(issue.code for issue in all_issues)),
        },
    }
    (args.out_dir / "semantic_gates_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_issue_csv(args.out_dir / "semantic_gate_issues.csv", draft_rows)
    _write_report_md(args.out_dir / "semantic_gates_report.md", report)
    return 0 if report["summary"]["semantic_pass"] else 2


def _find_answer_registry(bot_pack_dir: Path) -> Path | None:
    for name in ("answer_registry.json", "answer_registry.jsonl", "answer_registry.yaml", "answer_registry.yml"):
        path = bot_pack_dir / name
        if path.exists():
            return path
    return None


def _load_source_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return {str(item.get("source_id") or "") for item in payload if isinstance(item, Mapping)}
    if isinstance(payload, dict):
        records = payload.get("sources") or payload.get("records") or payload
        if isinstance(records, list):
            return {str(item.get("source_id") or "") for item in records if isinstance(item, Mapping)}
        if isinstance(records, dict):
            return set(str(key) for key in records)
    return set()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _read_jsonl_by_id(path: Path) -> dict[str, dict[str, Any]]:
    return {str(item.get("id") or item.get("test_id") or ""): item for item in _read_jsonl(path)}


def _write_issue_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "test_id",
        "code",
        "severity",
        "message",
        "brand",
        "priority",
        "category",
        "subcategory",
        "draft_text",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_report_md(path: Path, report: Mapping[str, Any]) -> None:
    summary = report["summary"]
    lines = [
        "# Semantic gates report",
        "",
        f"- formal_pass: `{summary['formal_pass']}`",
        f"- semantic_pass: `{summary['semantic_pass']}`",
        f"- answer_registry_entries: `{report['answer_registry_entries']}`",
        f"- registry_issue_count: `{summary['registry_issue_count']}`",
        f"- draft_issue_count: `{summary['draft_issue_count']}`",
        f"- errors: `{summary['errors']}`",
        f"- warnings: `{summary['warnings']}`",
        "",
        "## By code",
        "",
    ]
    for code, count in sorted(summary["by_code"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- `{code}`: {count}")
    if not report["answer_registry_path"]:
        lines.extend(
            [
                "",
                "## Note",
                "",
                "В bot pack пока нет `answer_registry.*`. Это ожидаемо для подготовительного этапа: каркас gates готов, но P1-реестр будет создан после ответов из опросника.",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
