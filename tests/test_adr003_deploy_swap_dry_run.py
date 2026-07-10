from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "adr003_deploy_swap_dry_run.py"


def run_helper(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )


def test_deploy_swap_dry_run_reads_freeze_from_live_worktree() -> None:
    result = run_helper(
        "--candidate-worktree",
        "/tmp/mango_candidate",
        "--live-worktree",
        "/tmp/mango_live",
        "--previous-worktree",
        "/tmp/mango_previous",
        "--previous-head",
        "eb6fa0b",
        "--previous-screen",
        "screen.previous",
    )

    assert result.returncode == 0, result.stderr
    assert "p='/tmp/mango_live/.codex_local/telegram_pilot_bots/runtime/public_pilot_bots_heartbeat.json'" in result.stdout
    assert 'cd "/tmp/mango_candidate"' in result.stdout
    assert 'cd "/tmp/mango_previous"' in result.stdout
    assert "# previous launch command:" in result.stdout
    assert "# previous profile: pilot_gold_v1" in result.stdout
    assert "# previous snapshot: v6.7 staging r4.1" in result.stdout


def test_deploy_swap_dry_run_refuses_same_candidate_and_previous() -> None:
    result = run_helper(
        "--candidate-worktree",
        "/tmp/mango_same",
        "--previous-worktree",
        "/tmp/mango_same",
        "--previous-head",
        "eb6fa0b",
        "--previous-screen",
        "screen.previous",
    )

    assert result.returncode == 2
    assert "previous-worktree equals candidate-worktree" in result.stderr
