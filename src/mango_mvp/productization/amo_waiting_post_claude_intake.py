from __future__ import annotations

import csv
import json
import shlex
import socket
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


AMO_WAITING_POST_CLAUDE_INTAKE_SCHEMA_VERSION = "amo_waiting_post_claude_intake_v1"
DEFAULT_AMO_WAITING_POST_CLAUDE_INTAKE_ROOT = Path("stable_runtime/amo_waiting_post_claude_intake_20260511_v1")

BLOCKING_SEVERITIES = {"P0", "P1", "P2"}


@dataclass(frozen=True)
class AmoWaitingPostClaudeIntakeSummary:
    schema_version: str
    generated_at: str
    result_dir: str
    waiting_root: str
    verdict: str
    status: str
    blocking_findings: int
    p0_findings: int
    p1_findings: int
    p2_findings: int
    p3_findings: int
    info_findings: int
    network_dry_run_allowed: bool
    live_write_allowed: bool
    tunnel_host: str
    tunnel_port: int
    tunnel_available: bool
    non_duplicate_candidate_rows: int
    refresh_candidate_rows: int
    readback_missing_rows: int
    contact_id_mismatch_rows: int

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


def build_amo_waiting_post_claude_intake(
    *,
    result_dir: Path,
    waiting_root: Path,
    out_root: Optional[Path] = None,
    generated_at: Optional[datetime] = None,
    tunnel_host: str = "127.0.0.1",
    tunnel_port: int = 15432,
    check_tunnel: bool = True,
) -> Mapping[str, Any]:
    """Convert Claude audit output into a deterministic next-stage gate.

    This does not run readback, dry-run, or live write. It only decides which
    follow-up commands may be offered to the operator.
    """

    result_dir = result_dir.expanduser().resolve(strict=False)
    waiting_root = waiting_root.expanduser().resolve(strict=False)
    out_root = out_root.expanduser().resolve(strict=False) if out_root else DEFAULT_AMO_WAITING_POST_CLAUDE_INTAKE_ROOT.resolve(strict=False)
    now = generated_at or datetime.now(timezone.utc)

    result_md = result_dir / "CLAUDE_REAUDIT_RESULT.md"
    findings_path = result_dir / "findings.csv"
    row_decisions_path = result_dir / "row_decisions.csv"
    waiting_summary_path = waiting_root / "summary.json"
    if not result_md.exists():
        raise FileNotFoundError(f"Claude result markdown not found: {result_md}")
    if not findings_path.exists():
        raise FileNotFoundError(f"Claude findings.csv not found: {findings_path}")
    if not row_decisions_path.exists():
        raise FileNotFoundError(f"Claude row_decisions.csv not found: {row_decisions_path}")
    if not waiting_summary_path.exists():
        raise FileNotFoundError(f"waiting summary not found: {waiting_summary_path}")

    result_text = result_md.read_text(encoding="utf-8")
    verdict = parse_verdict(result_text)
    findings = _read_csv(findings_path)
    row_decisions = _read_csv(row_decisions_path)
    severity_counts = Counter(str(row.get("severity") or "").strip().upper() for row in findings)
    status_counts = Counter(str(row.get("status") or "").strip().lower() for row in findings)
    decision_counts = Counter(str(row.get("decision") or "").strip().lower() for row in row_decisions)
    blocking_findings = sum(
        1
        for row in findings
        if str(row.get("severity") or "").strip().upper() in BLOCKING_SEVERITIES
        and str(row.get("status") or "").strip().lower() not in {"fixed", "accepted_risk", "closed"}
    )
    waiting_summary = _load_json(waiting_summary_path)
    waiting_counts = _mapping(waiting_summary.get("counts"))
    non_duplicate = _int(waiting_counts.get("non_duplicate_live_candidate_rows"))
    refresh = _int(waiting_counts.get("refresh_candidate_rows"))
    readback_missing = _int(waiting_counts.get("readback_missing_rows"))
    mismatch = _int(waiting_counts.get("contact_id_mismatch_rows"))
    tunnel_available = check_tcp_port(tunnel_host, tunnel_port) if check_tunnel else False
    network_allowed = verdict in {"PASS", "PASS_WITH_LIMITATIONS"} and blocking_findings == 0
    if network_allowed and (non_duplicate + refresh + readback_missing) == 0:
        network_allowed = False
    if not verdict:
        status = "blocked_missing_verdict"
    elif blocking_findings:
        status = "blocked_by_claude_findings"
    elif network_allowed and tunnel_available:
        status = "ready_for_real_tunnel_readback_and_dry_run"
    elif network_allowed:
        status = "waiting_for_shared_db_tunnel"
    else:
        status = "blocked_no_next_network_work"
    summary = AmoWaitingPostClaudeIntakeSummary(
        schema_version=AMO_WAITING_POST_CLAUDE_INTAKE_SCHEMA_VERSION,
        generated_at=now.isoformat(timespec="seconds"),
        result_dir=str(result_dir),
        waiting_root=str(waiting_root),
        verdict=verdict or "UNKNOWN",
        status=status,
        blocking_findings=blocking_findings,
        p0_findings=severity_counts.get("P0", 0),
        p1_findings=severity_counts.get("P1", 0),
        p2_findings=severity_counts.get("P2", 0),
        p3_findings=severity_counts.get("P3", 0),
        info_findings=severity_counts.get("INFO", 0),
        network_dry_run_allowed=network_allowed,
        live_write_allowed=False,
        tunnel_host=tunnel_host,
        tunnel_port=int(tunnel_port),
        tunnel_available=tunnel_available,
        non_duplicate_candidate_rows=non_duplicate,
        refresh_candidate_rows=refresh,
        readback_missing_rows=readback_missing,
        contact_id_mismatch_rows=mismatch,
    )
    payload = {
        "summary": summary.to_json_dict(),
        "finding_status_counts": dict(status_counts),
        "row_decision_counts": dict(decision_counts),
        "next_actions": build_next_actions(
            status=status,
            network_allowed=network_allowed,
            tunnel_available=tunnel_available,
            non_duplicate=non_duplicate,
            refresh=refresh,
            readback_missing=readback_missing,
            mismatch=mismatch,
        ),
        "inputs": {
            "result_md": str(result_md),
            "findings_csv": str(findings_path),
            "row_decisions_csv": str(row_decisions_path),
            "waiting_summary": str(waiting_summary_path),
        },
        "outputs": {
            "summary_json": str(out_root / "summary.json"),
            "command_center_md": str(out_root / "command_center.md"),
            "next_safe_network_commands_sh": str(out_root / "next_safe_network_commands.sh"),
        },
        "safety": {
            "read_only": True,
            "runs_network_dry_run": False,
            "runs_readback": False,
            "write_crm": False,
            "live_write_allowed": False,
            "requires_explicit_approval_before_live": True,
        },
    }
    if out_root:
        out_root.mkdir(parents=True, exist_ok=True)
        _write_json(out_root / "summary.json", payload)
        (out_root / "command_center.md").write_text(render_command_center(payload), encoding="utf-8")
        command_path = out_root / "next_safe_network_commands.sh"
        command_path.write_text(render_next_safe_network_commands(payload), encoding="utf-8")
        command_path.chmod(0o755)
    return payload


def parse_verdict(text: str) -> str:
    upper = text.upper()
    for verdict in ("PASS_WITH_LIMITATIONS", "PASS", "FAIL"):
        if verdict in upper:
            return verdict
    return ""


def build_next_actions(
    *,
    status: str,
    network_allowed: bool,
    tunnel_available: bool,
    non_duplicate: int,
    refresh: int,
    readback_missing: int,
    mismatch: int,
) -> list[Mapping[str, Any]]:
    actions: list[Mapping[str, Any]] = []
    if not network_allowed:
        return [
            {
                "action": "stop_and_review_claude_findings",
                "rows": 0,
                "allowed_now": False,
                "description_ru": "Есть блокирующие finding-и или нет PASS/PASS_WITH_LIMITATIONS.",
            }
        ]
    if not tunnel_available:
        actions.append(
            {
                "action": "bring_up_shared_db_tunnel",
                "rows": 0,
                "allowed_now": False,
                "description_ru": "Поднять AMO shared DB tunnel на 127.0.0.1:15432, затем повторить intake или запустить network commands.",
            }
        )
    if readback_missing:
        actions.append(
            {
                "action": "run_readback_missing",
                "rows": readback_missing,
                "allowed_now": tunnel_available,
                "description_ru": "Прочитать обратно 15 уже записанных строк; это read-only и prerequisite для refresh.",
            }
        )
    if non_duplicate:
        actions.append(
            {
                "action": "run_non_duplicate_real_tunnel_dry_run",
                "rows": non_duplicate,
                "allowed_now": tunnel_available,
                "description_ru": "Запустить real-tunnel dry-run по 1 non-duplicate кандидату; не live-write.",
            }
        )
    if refresh:
        actions.append(
            {
                "action": "run_refresh_real_tunnel_dry_run",
                "rows": refresh,
                "allowed_now": tunnel_available,
                "description_ru": "Запустить diff-based refresh dry-run по 40 already-written кандидатам; не live-write.",
            }
        )
    if mismatch:
        actions.append(
            {
                "action": "keep_contact_id_mismatch_blocked",
                "rows": mismatch,
                "allowed_now": False,
                "description_ru": "Contact-id mismatch остается blocked до ручной сверки/склейки.",
            }
        )
    actions.append(
        {
            "action": "live_write",
            "rows": 0,
            "allowed_now": False,
            "description_ru": "Live-write запрещен до нового audit pack, explicit approval и post-dry-run gates.",
        }
    )
    return actions


def render_command_center(payload: Mapping[str, Any]) -> str:
    summary = _mapping(payload.get("summary"))
    lines = [
        "# AMO waiting post-Claude command center",
        "",
        f"Generated at: `{summary.get('generated_at')}`",
        "",
        "## Status",
        "",
        f"- Claude verdict: `{summary.get('verdict')}`",
        f"- Intake status: `{summary.get('status')}`",
        f"- Blocking P0/P1/P2 findings: `{summary.get('blocking_findings')}`",
        f"- Network readback/dry-run allowed by audit: `{summary.get('network_dry_run_allowed')}`",
        f"- Shared DB tunnel available now: `{summary.get('tunnel_available')}`",
        f"- Live write allowed: `{summary.get('live_write_allowed')}`",
        "",
        "## Rows",
        "",
        f"- Non-duplicate dry-run candidates: `{summary.get('non_duplicate_candidate_rows')}`",
        f"- Refresh dry-run candidates: `{summary.get('refresh_candidate_rows')}`",
        f"- Readback-missing rows: `{summary.get('readback_missing_rows')}`",
        f"- Contact-id mismatch rows: `{summary.get('contact_id_mismatch_rows')}`",
        "",
        "## Next Actions",
        "",
    ]
    for action in payload.get("next_actions") or []:
        lines.append(
            f"- `{action['action']}`: allowed_now=`{action['allowed_now']}`, rows=`{action['rows']}`. {action['description_ru']}"
        )
    lines.extend(
        [
            "",
            "## Safe Network Commands",
            "",
            "These commands are readback/dry-run only. They do not live-write to AMO.",
            "",
            "```bash",
            "cd \"/Users/dmitrijfabarisov/Projects/Mango analyse\"",
            "stable_runtime/amo_waiting_post_claude_intake_20260511_v1/next_safe_network_commands.sh",
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def render_next_safe_network_commands(payload: Mapping[str, Any]) -> str:
    summary = _mapping(payload.get("summary"))
    waiting_root = Path(str(summary.get("waiting_root") or "stable_runtime/amo_waiting_autonomous_work_20260511_v1"))
    lines = [
        "#!/usr/bin/env bash",
        "set -uo pipefail",
        "",
        "cd \"$(dirname \"$0\")/../..\"",
        "",
        "if ! (echo > /dev/tcp/127.0.0.1/15432) >/dev/null 2>&1; then",
        "  echo \"BLOCKED: shared DB tunnel 127.0.0.1:15432 is not available.\" >&2",
        "  echo \"Start the AMO shared DB tunnel first, then rerun this script.\" >&2",
        "  exit 2",
        "fi",
        "",
        "echo \"Running readback/dry-run commands. No live-write flags are used.\"",
        "STEP_TIMEOUT_SECONDS=\"${AMO_SAFE_NETWORK_STEP_TIMEOUT_SECONDS:-360}\"",
        "FAILED_STEPS=0",
        "run_safe_step() {",
        "  local name=\"$1\"",
        "  local script_path=\"$2\"",
        "  echo \"=== ${name}: started; timeout=${STEP_TIMEOUT_SECONDS}s ===\"",
        "  if python3 - \"$STEP_TIMEOUT_SECONDS\" \"$script_path\" <<'PY'; then",
        "import subprocess",
        "import sys",
        "",
        "timeout = int(sys.argv[1])",
        "script_path = sys.argv[2]",
        "try:",
        "    completed = subprocess.run([\"bash\", script_path], timeout=timeout)",
        "except subprocess.TimeoutExpired:",
        "    print(f\"TIMEOUT: {script_path} exceeded {timeout}s\", file=sys.stderr)",
        "    raise SystemExit(124)",
        "raise SystemExit(completed.returncode)",
        "PY",
        "    echo \"=== ${name}: passed ===\"",
        "  else",
        "    local code=$?",
        "    FAILED_STEPS=$((FAILED_STEPS + 1))",
        "    echo \"=== ${name}: failed with exit code ${code}; continuing safe independent steps ===\" >&2",
        "  fi",
        "}",
    ]
    waiting_root_arg = shlex.quote(str(waiting_root))
    if _int(summary.get("readback_missing_rows")):
        lines.append(f"run_safe_step readback_missing {waiting_root_arg}/next_readback_missing_commands.sh")
    if _int(summary.get("non_duplicate_candidate_rows")):
        lines.append(f"run_safe_step non_duplicate_dry_run {waiting_root_arg}/next_non_duplicate_real_tunnel_dry_run_command.sh")
    if _int(summary.get("refresh_candidate_rows")):
        lines.append(f"run_safe_step refresh_dry_run {waiting_root_arg}/next_refresh_real_tunnel_dry_run_command.sh")
    lines.extend(
        [
            "",
            "if [ \"$FAILED_STEPS\" -ne 0 ]; then",
            "  echo \"Completed safe network commands with ${FAILED_STEPS} failed step(s). Live-write remains blocked.\" >&2",
            "  exit 1",
            "fi",
            "",
            "echo \"Done. Do not run live-write until the generated reports are audited and explicitly approved.\"",
        ]
    )
    return "\n".join(lines) + "\n"


def check_tcp_port(host: str, port: int, timeout: float = 0.4) -> bool:
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except OSError:
        return False


def _read_csv(path: Path) -> list[Mapping[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _load_json(path: Path) -> Mapping[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, Mapping) else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


__all__ = [
    "AMO_WAITING_POST_CLAUDE_INTAKE_SCHEMA_VERSION",
    "DEFAULT_AMO_WAITING_POST_CLAUDE_INTAKE_ROOT",
    "build_amo_waiting_post_claude_intake",
    "parse_verdict",
    "check_tcp_port",
]
