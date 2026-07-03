from __future__ import annotations

from pathlib import Path


def test_e3_runner_is_separate_and_has_expected_env_matrix() -> None:
    root = Path(__file__).resolve().parents[1]
    e3 = (root / "scripts/run_adr003_semantic_reading_e3_paired.sh").read_text(encoding="utf-8")
    e2 = (root / "scripts/run_adr003_semantic_reading_e2_triple.sh").read_text(encoding="utf-8")

    assert "TELEGRAM_RELIABLE_ANSWERER_STEP1=1" in e3
    assert "TELEGRAM_SEMANTIC_FRAME_SHADOW=1" in e3
    assert "TELEGRAM_SEMANTIC_READING_CLASSES=" in e3
    assert 'TELEGRAM_SEMANTIC_READING_CLASSES="$READING_CLASSES"' in e3
    assert "VALID_E3_{leg}" in e3
    assert "TELEGRAM_RELIABLE_ANSWERER_STEP1=1" not in e2
    assert "VALID_E3_" not in e2
