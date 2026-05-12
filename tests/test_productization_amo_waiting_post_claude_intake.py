from __future__ import annotations

import csv
import json
from pathlib import Path

from mango_mvp.productization.amo_waiting_post_claude_intake import (
    build_amo_waiting_post_claude_intake,
    parse_verdict,
)


def test_parse_verdict_prefers_pass_with_limitations() -> None:
    assert parse_verdict("Verdict: **PASS_WITH_LIMITATIONS**") == "PASS_WITH_LIMITATIONS"
    assert parse_verdict("Verdict: **PASS**") == "PASS"
    assert parse_verdict("Verdict: **FAIL**") == "FAIL"


def test_post_claude_intake_allows_network_dry_run_but_never_live(tmp_path: Path) -> None:
    result_dir, waiting_root = _fixture(tmp_path, verdict="PASS_WITH_LIMITATIONS", severities=["P3", "INFO"])
    out = tmp_path / "out"

    payload = build_amo_waiting_post_claude_intake(
        result_dir=result_dir,
        waiting_root=waiting_root,
        out_root=out,
        check_tunnel=False,
    )

    summary = payload["summary"]
    assert summary["network_dry_run_allowed"] is True
    assert summary["live_write_allowed"] is False
    assert summary["status"] == "waiting_for_shared_db_tunnel"
    assert summary["non_duplicate_candidate_rows"] == 1
    assert summary["refresh_candidate_rows"] == 40
    assert summary["readback_missing_rows"] == 15
    assert "live_write" in {action["action"] for action in payload["next_actions"]}
    script = (out / "next_safe_network_commands.sh").read_text(encoding="utf-8")
    assert "next_readback_missing_commands.sh" in script
    assert "'/tmp" in script or "'/" in script
    assert "Mango analyse" in script
    assert (out / "command_center.md").exists()


def test_post_claude_intake_blocks_on_p1_open_finding(tmp_path: Path) -> None:
    result_dir, waiting_root = _fixture(tmp_path, verdict="PASS_WITH_LIMITATIONS", severities=["P1"])

    payload = build_amo_waiting_post_claude_intake(
        result_dir=result_dir,
        waiting_root=waiting_root,
        out_root=tmp_path / "out",
        check_tunnel=False,
    )

    assert payload["summary"]["network_dry_run_allowed"] is False
    assert payload["summary"]["status"] == "blocked_by_claude_findings"


def _fixture(tmp_path: Path, *, verdict: str, severities: list[str]) -> tuple[Path, Path]:
    result_dir = tmp_path / "audits" / "_results" / "case"
    result_dir.mkdir(parents=True)
    (result_dir / "CLAUDE_REAUDIT_RESULT.md").write_text(f"Verdict: **{verdict}**\n", encoding="utf-8")
    _write_csv(
        result_dir / "findings.csv",
        [
            {
                "finding_id": f"F{idx}",
                "severity": severity,
                "class": "known_class",
                "status": "open",
                "row_id": "",
                "column": "",
                "evidence": "test",
                "recommendation": "test",
            }
            for idx, severity in enumerate(severities, start=1)
        ],
    )
    _write_csv(
        result_dir / "row_decisions.csv",
        [
            {"row_id": "non_duplicate", "decision": "allow", "reason": "ok", "known_class": "", "future_class_note": ""},
            {"row_id": "refresh", "decision": "allow", "reason": "ok", "known_class": "", "future_class_note": ""},
            {"row_id": "readback", "decision": "needs_review", "reason": "readback missing", "known_class": "", "future_class_note": ""},
            {"row_id": "mismatch", "decision": "block", "reason": "mismatch", "known_class": "", "future_class_note": ""},
        ],
    )
    waiting_root = tmp_path / "Mango analyse" / "stable_runtime" / "amo_waiting_autonomous_work_20260511_v1"
    waiting_root.mkdir(parents=True)
    (waiting_root / "summary.json").write_text(
        json.dumps(
            {
                "counts": {
                    "non_duplicate_live_candidate_rows": 1,
                    "refresh_candidate_rows": 40,
                    "readback_missing_rows": 15,
                    "contact_id_mismatch_rows": 1,
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return result_dir, waiting_root


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
