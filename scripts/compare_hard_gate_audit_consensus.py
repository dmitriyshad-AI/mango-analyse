#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

POLICY_GPT_ONLY = "gpt-only"
POLICY_CONSENSUS = "consensus"


def main() -> int:
    args = parse_args()
    package = args.package.resolve()
    audit_items_path = (args.audit_items or package / "audit_items.jsonl").resolve()
    gpt_path = (args.gpt_decisions or package / "gpt_audit_decisions.jsonl").resolve()
    claude_path = resolve_claude_decisions_path(package, args.claude_decisions)
    policy_mode = args.policy_mode
    out_dir = (args.out_dir or package).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not audit_items_path.exists():
        raise FileNotFoundError(f"audit_items.jsonl not found: {audit_items_path}")
    if not gpt_path.exists():
        raise FileNotFoundError(f"gpt_audit_decisions.jsonl not found: {gpt_path}")
    if policy_mode == POLICY_CONSENSUS and not claude_path.exists():
        raise FileNotFoundError(f"claude_decisions.jsonl not found: {claude_path}")

    audit_items = read_jsonl(audit_items_path)
    gpt_rows = read_jsonl(gpt_path)
    claude_rows = read_jsonl(claude_path) if claude_path.exists() else []
    mapping_rows = read_csv_if_exists(package / "PRIVATE_mapping_for_apply_do_not_edit.csv")
    safeguards_rows = read_csv_if_exists(package / "gpt_after_safeguards_200.csv")
    task_to_audit = {str(item.get("task_id")): str(item.get("audit_id")) for item in audit_items}
    item_by_audit = {str(item.get("audit_id")): item for item in audit_items}
    gpt_by_audit = {str(row.get("audit_id")): row for row in gpt_rows}
    claude_by_audit = {audit_id(row, task_to_audit): row for row in claude_rows}
    claude_by_audit.pop("", None)
    mapping_by_audit = {str(row.get("audit_id") or ""): row for row in mapping_rows if str(row.get("audit_id") or "")}
    safeguards_by_audit = {str(row.get("audit_id") or ""): row for row in safeguards_rows if str(row.get("audit_id") or "")}

    comparison: list[dict[str, Any]] = []
    apply_plan: list[dict[str, Any]] = []
    blocked_apply_plan: list[dict[str, Any]] = []
    non_auto_analysis: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    policy_counts: Counter[str] = Counter()
    gpt_counts: Counter[str] = Counter()
    claude_counts: Counter[str] = Counter()
    missing_gpt: list[str] = []
    missing_claude: list[str] = []

    for item in audit_items:
        aid = str(item.get("audit_id"))
        gpt = gpt_by_audit.get(aid)
        claude = claude_by_audit.get(aid)
        if gpt is None:
            missing_gpt.append(aid)
        if claude is None and policy_mode == POLICY_CONSENSUS:
            missing_claude.append(aid)
        gpt_decision = normalize_decision(gpt or {}, prefix="") if gpt else "missing"
        claude_decision = normalize_decision(claude or {}, prefix="claude_") if claude else "not_used"
        queue = consensus_queue(gpt_decision, claude_decision)
        policy_queue_value = policy_queue(policy_mode, queue, gpt_decision)
        safeguard = safeguards_by_audit.get(aid, {})
        mapping = mapping_by_audit.get(aid, {})
        policy = policy_decision(policy_mode, queue, policy_queue_value, gpt_decision, claude_decision, safeguard)
        counts[queue] += 1
        policy_counts[policy_queue_value] += 1
        gpt_counts[gpt_decision] += 1
        claude_counts[claude_decision] += 1
        comparison_row = (
            {
                "audit_id": aid,
                "task_id": item.get("task_id", ""),
                "source_filename": ((item.get("call") or {}).get("source_filename") or ""),
                "month": ((item.get("stratum") or {}).get("month") or ""),
                "current_call_type": ((item.get("stratum") or {}).get("current_call_type") or ""),
                "subtype": ((item.get("stratum") or {}).get("recommended_contact_subtype") or ""),
                "gpt_decision": gpt_decision,
                "gpt_confidence": (gpt or {}).get("confidence", ""),
                "claude_decision": claude_decision,
                "claude_confidence": (claude or {}).get("confidence", (claude or {}).get("claude_confidence", "")),
                "consensus_queue": queue,
                "policy_mode": policy_mode,
                "policy_queue": policy_queue_value,
                "safeguard_decision": safeguard.get("new_decision", ""),
                "safeguard_label": safeguard.get("new_label", ""),
                "safeguard_score": safeguard.get("new_score", ""),
                "safeguard_reason_codes": safeguard.get("reason_codes", ""),
                "policy_auto_apply_allowed": policy["allowed"],
                "policy_blockers": "|".join(policy["blockers"]),
                "gpt_reason": (gpt or {}).get("reason_ru", ""),
                "claude_reason": (claude or {}).get("reason_ru", (claude or {}).get("claude_reason", "")),
            }
        )
        comparison.append(comparison_row)
        plan_row = apply_plan_row(
            item=item,
            comparison=comparison_row,
            mapping=mapping,
            safeguard=safeguard,
            policy=policy,
            policy_mode=policy_mode,
        )
        if policy["allowed"]:
            apply_plan.append(plan_row)
        else:
            blocked_apply_plan.append(plan_row)
            non_auto_analysis.append(non_auto_row(comparison_row, plan_row))

    summary = {
        "package": str(package),
        "policy_mode": policy_mode,
        "audit_items": len(audit_items),
        "gpt_rows": len(gpt_rows),
        "claude_rows": len(claude_rows),
        "audit_items_path": str(audit_items_path),
        "gpt_decisions_path": str(gpt_path),
        "claude_decisions_path": str(claude_path),
        "missing_gpt": missing_gpt,
        "missing_claude": missing_claude,
        "counts_by_gpt_decision": dict(gpt_counts),
        "counts_by_claude_decision": dict(claude_counts),
        "counts_by_consensus_queue": dict(counts),
        "counts_by_policy_queue": dict(policy_counts),
        "strict_consensus_auto_apply": counts.get("consensus_auto_apply", 0),
        "needs_review_or_reanalyze": len(blocked_apply_plan),
        "policy_auto_apply_allowed": len(apply_plan),
        "policy_blocked": len(blocked_apply_plan),
        "non_auto_safeguard_decision_counts": dict(Counter(row["safeguard_decision"] for row in non_auto_analysis).most_common()),
    }

    auto_apply = [row for row in comparison if row["policy_auto_apply_allowed"]]
    review_queue = [row for row in comparison if not row["policy_auto_apply_allowed"]]
    keep_current = [row for row in comparison if row["policy_queue"] in {"consensus_keep_current", "gpt_keep_current"}]
    if policy_mode == POLICY_GPT_ONLY:
        write_csv(out_dir / "audit_gpt_comparison.csv", comparison)
        write_csv(out_dir / "audit_gpt_auto_apply_candidates.csv", auto_apply)
        write_csv(out_dir / "audit_gpt_review_queue.csv", review_queue)
        write_csv(out_dir / "audit_gpt_keep_current.csv", keep_current)
        write_csv(out_dir / "audit_gpt_apply_plan.csv", apply_plan)
        write_csv(out_dir / "audit_gpt_blocked_apply_plan.csv", blocked_apply_plan)
        write_csv(out_dir / "audit_gpt_non_auto_analysis.csv", non_auto_analysis)
        (out_dir / "audit_gpt_policy.json").write_text(json.dumps(policy_json(summary), ensure_ascii=False, indent=2), encoding="utf-8")
        (out_dir / "audit_gpt_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        (out_dir / "audit_gpt_report.md").write_text(markdown_report(summary), encoding="utf-8")
    else:
        write_csv(out_dir / "audit_consensus_comparison.csv", comparison)
        write_csv(out_dir / "audit_consensus_auto_apply_candidates.csv", auto_apply)
        write_csv(out_dir / "audit_consensus_review_queue.csv", review_queue)
        write_csv(out_dir / "audit_consensus_keep_current.csv", keep_current)
        write_csv(out_dir / "audit_consensus_apply_plan.csv", apply_plan)
        write_csv(out_dir / "audit_consensus_blocked_apply_plan.csv", blocked_apply_plan)
        write_csv(out_dir / "audit_consensus_non_auto_analysis.csv", non_auto_analysis)
        (out_dir / "audit_consensus_policy.json").write_text(json.dumps(policy_json(summary), ensure_ascii=False, indent=2), encoding="utf-8")
        (out_dir / "audit_consensus_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        (out_dir / "audit_consensus_report.md").write_text(markdown_report(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def resolve_claude_decisions_path(package: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.resolve()
    candidates = [
        package / "claude_decisions.jsonl",
        package / "CLAUDE answer" / "claude_decisions.jsonl",
        package / "claude_answer" / "claude_decisions.jsonl",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def read_csv_if_exists(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def audit_id(row: dict[str, Any], task_to_audit: dict[str, str]) -> str:
    explicit = str(row.get("audit_id") or "")
    if explicit:
        return explicit
    task_id = str(row.get("task_id") or "")
    return task_to_audit.get(task_id, "")


def normalize_decision(row: dict[str, Any], *, prefix: str) -> str:
    raw = str(row.get("decision") or row.get(f"{prefix}decision") or "").strip()
    if raw in {"safe_apply", "force_non_conversation"}:
        return "safe_apply"
    if raw in {"keep_current", "keep_current_analysis"}:
        return "keep_current"
    if raw in {"manual_review", "human_review_required"}:
        return "manual_review"
    if raw == "reanalyze_required":
        return "reanalyze_required"
    return raw or "unknown"


def consensus_queue(gpt: str, claude: str) -> str:
    if claude == "not_used":
        return "claude_not_used"
    if "missing" in {gpt, claude}:
        return "blocked_missing_decision"
    if gpt == claude == "safe_apply":
        return "consensus_auto_apply"
    if gpt == claude == "keep_current":
        return "consensus_keep_current"
    if "reanalyze_required" in {gpt, claude}:
        return "reanalyze_required"
    if "manual_review" in {gpt, claude}:
        return "manual_review"
    return "disagreement_review"


def policy_queue(policy_mode: str, consensus: str, gpt: str) -> str:
    if policy_mode == POLICY_CONSENSUS:
        return consensus
    if gpt == "safe_apply":
        return "gpt_auto_apply"
    if gpt == "keep_current":
        return "gpt_keep_current"
    if gpt == "manual_review":
        return "gpt_manual_review"
    if gpt == "reanalyze_required":
        return "reanalyze_required"
    return f"gpt_blocked_{gpt}"


def policy_decision(
    policy_mode: str,
    queue: str,
    policy_queue_value: str,
    gpt: str,
    claude: str,
    safeguard: dict[str, str],
) -> dict[str, Any]:
    blockers: list[str] = []
    if gpt != "safe_apply":
        blockers.append(f"gpt_decision:{gpt}")
    if policy_mode == POLICY_CONSENSUS:
        if queue != "consensus_auto_apply":
            blockers.append(f"consensus_queue:{queue}")
        if claude != "safe_apply":
            blockers.append(f"claude_decision:{claude}")
    elif policy_queue_value != "gpt_auto_apply":
        blockers.append(f"policy_queue:{policy_queue_value}")
    if safeguard:
        if safeguard.get("new_decision") != "safe_apply":
            blockers.append(f"safeguard_decision:{safeguard.get('new_decision') or 'missing'}")
        if safeguard.get("new_label") != "non_conversation_high_confidence":
            blockers.append(f"safeguard_label:{safeguard.get('new_label') or 'missing'}")
    return {"allowed": not blockers, "blockers": blockers}


def apply_plan_row(
    *,
    item: dict[str, Any],
    comparison: dict[str, Any],
    mapping: dict[str, str],
    safeguard: dict[str, str],
    policy: dict[str, Any],
    policy_mode: str,
) -> dict[str, Any]:
    call = item.get("call") or {}
    stratum = item.get("stratum") or {}
    current_state = item.get("current_state") or {}
    analysis = current_state.get("analysis") if isinstance(current_state.get("analysis"), dict) else {}
    current_contentful = str((analysis.get("call_type") or stratum.get("current_call_type")) != "non_conversation")
    audit_id_value = str(item.get("audit_id") or comparison.get("audit_id") or "")
    review_hash = hashlib.sha256(
        "|".join(
            [
                f"hard_gate_{policy_mode}_v1",
                audit_id_value,
                str(comparison.get("gpt_decision") or ""),
                str(comparison.get("claude_decision") or ""),
                str(comparison.get("safeguard_decision") or ""),
                str(comparison.get("safeguard_reason_codes") or ""),
            ]
        ).encode("utf-8")
    ).hexdigest()[:16]
    return {
        "audit_id": audit_id_value,
        "task_id": item.get("task_id", ""),
        "db": mapping.get("db") or call.get("db", ""),
        "id": mapping.get("call_record_id") or call.get("call_record_id", ""),
        "source_filename": mapping.get("source_filename") or call.get("source_filename", ""),
        "phone": mapping.get("phone") or call.get("phone", ""),
        "started_at": mapping.get("started_at") or call.get("started_at", ""),
        "manager_name": mapping.get("manager_name") or call.get("manager_name", ""),
        "duration_sec": call.get("duration_sec", ""),
        "current_call_type": stratum.get("current_call_type", ""),
        "current_contentful": current_contentful,
        "guardrail_label": safeguard.get("new_label", ""),
        "guardrail_score": safeguard.get("new_score", ""),
        "guardrail_reason_codes": safeguard.get("reason_codes", ""),
        "should_force_non_conversation": str(safeguard.get("new_decision") == "safe_apply"),
        "recommended_contact_subtype": stratum.get("recommended_contact_subtype", ""),
        "review_decision": review_decision(policy_mode, bool(policy["allowed"])),
        "review_hash": review_hash,
        "policy_mode": policy_mode,
        "policy_queue": comparison.get("policy_queue", ""),
        "consensus_queue": comparison.get("consensus_queue", ""),
        "gpt_decision": comparison.get("gpt_decision", ""),
        "claude_decision": comparison.get("claude_decision", ""),
        "policy_auto_apply_allowed": policy["allowed"],
        "policy_blockers": "|".join(policy["blockers"]),
        "gpt_reason": comparison.get("gpt_reason", ""),
        "claude_reason": comparison.get("claude_reason", ""),
    }


def review_decision(policy_mode: str, allowed: bool) -> str:
    if policy_mode == POLICY_GPT_ONLY:
        return "hard_gate_gpt_auto_apply" if allowed else "blocked_by_hard_gate_gpt_policy"
    return "hard_gate_consensus_auto_apply" if allowed else "blocked_by_hard_gate_consensus"


def non_auto_row(comparison: dict[str, Any], plan_row: dict[str, Any]) -> dict[str, Any]:
    return {
        "audit_id": comparison.get("audit_id", ""),
        "source_filename": comparison.get("source_filename", ""),
        "current_call_type": comparison.get("current_call_type", ""),
        "subtype": comparison.get("subtype", ""),
        "gpt_decision": comparison.get("gpt_decision", ""),
        "claude_decision": comparison.get("claude_decision", ""),
        "consensus_queue": comparison.get("consensus_queue", ""),
        "policy_mode": comparison.get("policy_mode", ""),
        "policy_queue": comparison.get("policy_queue", ""),
        "safeguard_decision": comparison.get("safeguard_decision", ""),
        "safeguard_label": comparison.get("safeguard_label", ""),
        "safeguard_reason_codes": comparison.get("safeguard_reason_codes", ""),
        "policy_blockers": plan_row.get("policy_blockers", ""),
        "recommended_handling": recommended_handling(str(comparison.get("consensus_queue") or "")),
        "gpt_reason": comparison.get("gpt_reason", ""),
        "claude_reason": comparison.get("claude_reason", ""),
    }


def recommended_handling(queue: str) -> str:
    if queue == "consensus_keep_current":
        return "keep_current_do_not_apply"
    if queue == "manual_review":
        return "manual_or_llm_review_do_not_apply"
    if queue == "disagreement_review":
        return "disagreement_review_do_not_apply"
    return "blocked_do_not_apply"


def policy_json(summary: dict[str, Any]) -> dict[str, Any]:
    policy_mode = str(summary.get("policy_mode") or POLICY_GPT_ONLY)
    if policy_mode == POLICY_GPT_ONLY:
        auto_apply_requires = [
            "gpt_decision == safe_apply",
            "deterministic_safeguard_decision == safe_apply",
            "deterministic_safeguard_label == non_conversation_high_confidence",
        ]
        blocked_queues = ["gpt_keep_current", "gpt_manual_review", "reanalyze_required", "gpt_blocked_*"]
        version = "hard_gate_gpt_policy_v1"
    else:
        auto_apply_requires = [
            "consensus_queue == consensus_auto_apply",
            "gpt_decision == safe_apply",
            "claude_decision == safe_apply",
            "deterministic_safeguard_decision == safe_apply",
            "deterministic_safeguard_label == non_conversation_high_confidence",
        ]
        blocked_queues = ["consensus_keep_current", "manual_review", "disagreement_review", "reanalyze_required"]
        version = "hard_gate_consensus_policy_v1"
    return {
        "policy_version": version,
        "policy_mode": policy_mode,
        "created_from": summary.get("package"),
        "auto_apply_requires": auto_apply_requires,
        "blocked_queues": blocked_queues,
        "strict_consensus_auto_apply": summary.get("strict_consensus_auto_apply"),
        "policy_auto_apply_allowed": summary.get("policy_auto_apply_allowed"),
        "needs_review_or_reanalyze": summary.get("needs_review_or_reanalyze"),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Hard Gate Audit Consensus Report",
        "",
        f"Package: `{summary['package']}`",
        f"Policy mode: `{summary['policy_mode']}`",
        "",
        f"- Audit items: `{summary['audit_items']}`",
        f"- GPT rows: `{summary['gpt_rows']}`",
        f"- Claude rows: `{summary['claude_rows']}`",
        f"- GPT decisions: `{summary['counts_by_gpt_decision']}`",
        f"- Claude decisions: `{summary['counts_by_claude_decision']}`",
        "",
        "## Consensus Queues",
        "",
    ]
    for key, value in summary["counts_by_consensus_queue"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Policy Queues", ""])
    for key, value in summary["counts_by_policy_queue"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `audit_consensus_comparison.csv`",
            "- `audit_consensus_auto_apply_candidates.csv`",
            "- `audit_consensus_review_queue.csv`",
            "- `audit_consensus_keep_current.csv`",
            "- `audit_consensus_apply_plan.csv`",
            "- `audit_consensus_blocked_apply_plan.csv`",
            "- `audit_consensus_non_auto_analysis.csv`",
            "- `audit_consensus_policy.json`",
            "- `audit_consensus_summary.json`",
            "- `audit_consensus_report.md`",
            "- `audit_gpt_apply_plan.csv` if policy mode is `gpt-only`",
            "- `audit_gpt_blocked_apply_plan.csv` if policy mode is `gpt-only`",
            "- `audit_gpt_policy.json` if policy mode is `gpt-only`",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build hard-gate audit apply plans from GPT and optional Claude decisions.")
    parser.add_argument(
        "--package",
        type=Path,
        default=Path("stable_runtime/non_conversation_hard_gate_audit_package_200_20260509"),
    )
    parser.add_argument("--audit-items", type=Path, default=None)
    parser.add_argument("--gpt-decisions", type=Path, default=None)
    parser.add_argument("--claude-decisions", type=Path, default=None)
    parser.add_argument("--policy-mode", choices=[POLICY_GPT_ONLY, POLICY_CONSENSUS], default=POLICY_GPT_ONLY)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
