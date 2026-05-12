from __future__ import annotations

import json
from pathlib import Path

from mango_mvp.quality.bot_safety_frozen_corpus import (
    BotSafetyCorpusValidationConfig,
    BotSafetyFrozenCorpusConfig,
    build_bot_safety_frozen_corpus,
    validate_bot_safety_frozen_corpus,
    validate_one_case,
)


def test_validate_one_case_sanitizes_and_enforces_idempotence() -> None:
    case = {
        "case_id": "unit-1",
        "layer": "synthetic",
        "risk_class": "money",
        "severity": "P0",
        "input_text": "Первый семестр за 88000, год целиком за 147000.",
        "forbidden_patterns": json.dumps(["88000", "147000"], ensure_ascii=False),
    }

    result = validate_one_case(case, detector_min_severity="P2")

    assert result["passed"] == "yes"
    assert result["strong_idempotence"] == "yes"
    assert result["forbidden_hits"] == ""
    assert result["detector_findings"] == 0


def test_frozen_corpus_builder_and_validator_smoke(tmp_path: Path) -> None:
    corpus_root = tmp_path / "corpus"
    validation_root = tmp_path / "validation"

    build_summary = build_bot_safety_frozen_corpus(
        BotSafetyFrozenCorpusConfig(
            out_root=corpus_root,
            synthetic_target=80,
            real_sample_size=0,
        )
    )
    validate_summary = validate_bot_safety_frozen_corpus(
        BotSafetyCorpusValidationConfig(
            corpus_jsonl=Path(build_summary["outputs"]["corpus_jsonl"]),
            out_root=validation_root,
            detector_min_severity="P2",
        )
    )

    assert build_summary["rows"] >= 80
    assert validate_summary["rows"] == build_summary["rows"]
    assert validate_summary["passed"] is True
    assert validate_summary["failures"] == 0
    assert (validation_root / "validation_results.csv").exists()
