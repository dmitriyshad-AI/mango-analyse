from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_adr003_flag_acceptance_pair.sh"


def _runner_text() -> str:
    return RUNNER.read_text(encoding="utf-8")


def test_adr003_flag_acceptance_runner_shell_syntax() -> None:
    subprocess.run(["bash", "-n", str(RUNNER)], check=True)


def test_adr003_flag_acceptance_runner_allows_only_package_flags() -> None:
    text = _runner_text()

    assert "TELEGRAM_TEXT_HYGIENE_PAYMENT_FIX|TELEGRAM_DIALOG_SUMMARY_ROLLING|TELEGRAM_INTENT_MODEL_LED" in text
    assert "TARGET_FLAG must be one of" in text


def test_adr003_flag_acceptance_runner_isolates_sibling_package_flags() -> None:
    text = _runner_text()

    assert "-u TELEGRAM_TEXT_HYGIENE_PAYMENT_FIX" in text
    assert "-u TELEGRAM_DIALOG_SUMMARY_ROLLING" in text
    assert "-u TELEGRAM_INTENT_MODEL_LED" in text
    assert "package_flags_off=(" in text
    assert "TELEGRAM_TEXT_HYGIENE_PAYMENT_FIX=0" in text
    assert "TELEGRAM_DIALOG_SUMMARY_ROLLING=0" in text
    assert "TELEGRAM_INTENT_MODEL_LED=0" in text
    assert "-u TELEGRAM_SEMANTIC_READING_CLASSES" in text
    assert "TELEGRAM_DIRECT_PATH_PILOT_CONFIG=pilot_gold_v1" in text
    assert "--allow-non-pilot-profile" in text


def test_adr003_flag_acceptance_runner_requires_model_led_application() -> None:
    text = _runner_text()

    assert "set -euo pipefail" in text
    assert '[[ "$TARGET_FLAG" == "TELEGRAM_INTENT_MODEL_LED" ]]' in text
    assert "--require-intent-model-led-application" in text
    assert 'run_leg B "${base_env[@]}" "${package_flags_off[@]}"' in text
    assert '"$TARGET_FLAG=$TARGET_FLAG_VALUE"' in text


def test_adr003_flag_acceptance_runner_validates_traced_profile_baseline() -> None:
    text = _runner_text()

    assert "--expect-trace" in text
    assert "--progress-json \"$OUT/$leg/progress.json\"" in text
    assert "--progress-leg \"$leg\"" in text
    assert "report_adr003_semantic_frame_eval.py" in text
    assert "sha_manifest.json" in text
    assert 'run_report || report_rc=$?' in text
    assert text.rindex('run_report || report_rc=$?') < text.rindex("write_sha_manifest")
    assert 'exit "$report_rc"' in text
