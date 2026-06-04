#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.derive_kb_schedule_2026_27_sources import BRAND_TO_SOURCE, build_schedule_fact


DEFAULT_RELEASE_DIR = PROJECT_ROOT / "product_data" / "knowledge_base" / "kb_release_20260603_v6_5_summer_format_cleanup"
DEFAULT_SCHEDULE_PACK = PROJECT_ROOT / "audits" / "_inbox" / "schedule_vs_tallanto_recheck_20260602_123118"
LEAK_MARKERS = (
    "Tallanto",
    "source_id",
    "fact_id",
    "freshness_status",
    "match_key",
    "FFFF0000",
    "theme:1",
    "course_number_seats",
    "места есть",
    "гарант",
    "записали",
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify KB schedule 2026/27 facts in a built release.")
    parser.add_argument("--release-dir", type=Path, default=DEFAULT_RELEASE_DIR)
    parser.add_argument("--schedule-pack", type=Path, default=DEFAULT_SCHEDULE_PACK)
    parser.add_argument("--ingest-decision", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    report = verify_schedule_release(
        release_dir=args.release_dir,
        schedule_pack=args.schedule_pack,
        ingest_decision=args.ingest_decision,
    )
    out_dir = args.out_dir.expanduser().resolve(strict=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "schedule_release_gate.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (out_dir / "schedule_release_gate.md").write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


def verify_schedule_release(*, release_dir: Path, schedule_pack: Path, ingest_decision: Path) -> dict[str, Any]:
    release_root = release_dir.expanduser().resolve(strict=False)
    schedule_root = schedule_pack.expanduser().resolve(strict=False)
    facts = load_release_facts(release_root)
    schedule_facts = {
        str(fact.get("fact_key") or ""): fact
        for fact in facts
        if str(fact.get("fact_key") or "").startswith("schedule_2026_27.groups.")
    }
    comparison_payload = load_json(schedule_root / "schedule_vs_tallanto_comparison.json")
    comparisons = comparison_payload.get("comparisons")
    if not isinstance(comparisons, list):
        raise ValueError("Missing comparisons list.")
    tallanto_rows = load_json(schedule_root / "tallanto_schedule_normalized.json")
    if not isinstance(tallanto_rows, list):
        raise ValueError("Missing Tallanto normalized list.")
    tallanto_by_key = {str(row.get("match_key") or ""): row for row in tallanto_rows if isinstance(row, Mapping)}
    ingest_rows = list(csv.DictReader(ingest_decision.expanduser().open(encoding="utf-8")))

    findings: list[dict[str, str]] = []
    expected_count_by_brand: Counter[str] = Counter()
    matched_expected = 0
    for row in comparisons:
        if not isinstance(row, Mapping) or str(row.get("status") or "") != "совпало":
            continue
        tallanto_row = tallanto_by_key.get(str(row.get("match_key") or "")) or {}
        matches = row.get("tallanto_matches") if isinstance(row.get("tallanto_matches"), list) else []
        match = matches[0] if matches and isinstance(matches[0], Mapping) else {}
        expected = build_schedule_fact(row, tallanto_row, match)
        fact_key = f"schedule_2026_27.groups.{expected['key']}.client_safe_text"
        expected_brand = BRAND_TO_SOURCE[str(row.get("brand") or "")]
        expected_count_by_brand[expected_brand] += 1
        fact = schedule_facts.get(fact_key)
        if not fact:
            findings.append(finding("P1", "missing_schedule_fact", fact_key, "Expected schedule fact is absent."))
            continue
        matched_expected += 1
        if str(fact.get("brand") or "") != expected_brand:
            findings.append(finding("P1", "wrong_brand", fact_key, f"Expected {expected_brand}, got {fact.get('brand')!r}."))
        if not fact.get("allowed_for_client_answer"):
            findings.append(finding("P1", "not_client_allowed", fact_key, "Schedule fact is not client-allowed."))
        if str(fact.get("client_safe_text") or "") != expected["client_safe_text"]:
            findings.append(finding("P1", "client_safe_text_mismatch", fact_key, str(fact.get("client_safe_text") or "")[:240]))
        expected_valid_until = str(match.get("date_finish") or tallanto_row.get("date_finish") or "")
        if str(fact.get("valid_until") or "") != expected_valid_until:
            findings.append(
                finding(
                    "P1",
                    "valid_until_mismatch",
                    fact_key,
                    f"Expected {expected_valid_until}, got {fact.get('valid_until')!r}.",
                )
            )

    for fact in schedule_facts.values():
        text = str(fact.get("client_safe_text") or "")
        fact_key = str(fact.get("fact_key") or "")
        for marker in LEAK_MARKERS:
            if marker.casefold() in text.casefold():
                findings.append(finding("P1", "client_text_leak_or_unsafe_claim", fact_key, marker))
        brand = str(fact.get("brand") or "")
        if brand == "foton" and re.search(r"\b(УНПК|АНО ДПО|Сретенка)", text, re.I):
            findings.append(finding("P1", "foton_schedule_cross_brand_text", fact_key, text[:240]))
        if brand == "unpk" and re.search(r"\b(Фотон|ЦДПО|ЦРДО)", text, re.I):
            findings.append(finding("P1", "unpk_schedule_cross_brand_text", fact_key, text[:240]))

    for fact in facts:
        if not fact.get("allowed_for_client_answer"):
            continue
        key = str(fact.get("fact_key") or "")
        text = str(fact.get("client_safe_text") or "")
        lowered = text.casefold().replace("ё", "е")
        if "пн-вс 10:00-18:00" in lowered and "расписан" in lowered:
            findings.append(finding("P1", "contact_hours_as_schedule", key, text[:240]))
        if re.search(r"расписан\w*\s+(?:и\s+подробная\s+информация\s+)?появ", lowered):
            findings.append(finding("P1", "stale_schedule_publication_text", key, text[:240]))

    ingest_decisions = Counter(row.get("decision", "") for row in ingest_rows)
    schedule_count_by_brand = Counter(str(fact.get("brand") or "") for fact in schedule_facts.values())
    summary = {
        "schedule_facts": len(schedule_facts),
        "schedule_facts_by_brand": dict(schedule_count_by_brand),
        "expected_by_brand": dict(expected_count_by_brand),
        "matched_expected": matched_expected,
        "ingest_rows": len(ingest_rows),
        "ingest_decisions": dict(ingest_decisions),
    }
    if len(schedule_facts) != 107:
        findings.append(finding("P1", "wrong_schedule_fact_count", "schedule_2026_27", str(len(schedule_facts))))
    if dict(schedule_count_by_brand) != {"foton": 45, "unpk": 62}:
        findings.append(finding("P1", "wrong_schedule_brand_counts", "schedule_2026_27", str(dict(schedule_count_by_brand))))
    if ingest_decisions != Counter({"include": 107}):
        findings.append(finding("P1", "wrong_ingest_decisions", str(ingest_decision), str(dict(ingest_decisions))))

    blocking = [item for item in findings if item["severity"] in {"P0", "P1"}]
    return {
        "schema_version": "kb_schedule_release_gate_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "release_dir": str(release_root),
        "schedule_pack": str(schedule_root),
        "ingest_decision": str(ingest_decision),
        "passed": not blocking,
        "blocking_findings": len(blocking),
        "summary": summary,
        "findings": findings,
    }


def load_release_facts(release_dir: Path) -> list[Mapping[str, Any]]:
    snapshot = load_json(release_dir / "kb_release_v3_snapshot.json")
    facts = snapshot.get("facts") or snapshot.get("facts_registry") or []
    if not isinstance(facts, list):
        raise ValueError("Release snapshot does not contain facts list.")
    return [fact for fact in facts if isinstance(fact, Mapping)]


def finding(severity: str, check_id: str, item_id: str, evidence: str) -> dict[str, str]:
    return {
        "severity": severity,
        "check_id": check_id,
        "item_id": item_id,
        "evidence": evidence,
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    summary = report.get("summary") if isinstance(report.get("summary"), Mapping) else {}
    lines = [
        "# Schedule Release Gate",
        "",
        f"- passed: `{report.get('passed')}`",
        f"- blocking_findings: `{report.get('blocking_findings')}`",
        f"- schedule_facts: `{summary.get('schedule_facts')}`",
        f"- schedule_facts_by_brand: `{summary.get('schedule_facts_by_brand')}`",
        f"- ingest_decisions: `{summary.get('ingest_decisions')}`",
        "",
        "## Findings",
        "",
    ]
    findings = report.get("findings") if isinstance(report.get("findings"), list) else []
    if not findings:
        lines.append("- None.")
    for item in findings:
        if not isinstance(item, Mapping):
            continue
        lines.append(
            f"- `{item.get('severity')}` `{item.get('check_id')}` `{item.get('item_id')}` — {item.get('evidence')}"
        )
    return "\n".join(lines).rstrip() + "\n"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
