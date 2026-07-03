from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_adr003_semantic_reading_e2_triple.sh"


def _runner_text() -> str:
    return RUNNER.read_text(encoding="utf-8")


def test_adr003_e2_runner_shell_syntax() -> None:
    subprocess.run(["bash", "-n", str(RUNNER)], check=True)


def test_adr003_e2_runner_forces_live_direct_path_profile() -> None:
    text = _runner_text()

    assert "TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1" in text
    assert "TELEGRAM_DIRECT_PATH=1" in text
    assert "TELEGRAM_BOT_GOLD_REAL=1" in text
    assert "TELEGRAM_TEMPLATE_FROM_KB=1" in text
    assert "TELEGRAM_ROUTE_RUBRIC=1" in text
    assert "TELEGRAM_LLM_RETRIEVE=1" in text
    assert "TELEGRAM_SEMANTIC_READING_CLASSES=" in text


def test_adr003_e2_runner_has_fail_fast_direct_path_and_frame_checks() -> None:
    text = _runner_text()

    assert "validate_direct_leg B" in text
    assert "validate_direct_leg I" in text
    assert "validate_inline_frame_leg I" in text
    assert "bot_direct_draft" in text
    assert "bot_direct_path" in text
    assert "bot_semantic_frame_shadow" in text
    assert "bot_semantic_frame" in text
    assert "pilot profile not active" in text


def test_adr003_e2_runner_exposes_dry_check_and_manifest() -> None:
    text = _runner_text()

    assert "--dry-check" in text
    assert "--limit 2" in text
    assert "sha_manifest.json" in text
    assert "schema_version" in text
    assert "adr003_semantic_reading_e2_triple_v2" in text
