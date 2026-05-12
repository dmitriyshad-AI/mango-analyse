from __future__ import annotations

from pathlib import Path

from mango_mvp.quality.crm_writeback_frozen_corpus import (
    CrmWritebackCorpusValidationConfig,
    validate_crm_writeback_frozen_corpus,
    validate_one_case,
)


FIXTURE = Path("tests/fixtures/crm_writeback_relevance_frozen_corpus.jsonl")


def test_validate_one_case_blocks_out_of_domain_context() -> None:
    row = validate_one_case(
        {
            "case_id": "x",
            "expected_decision": "block",
            "input_text": "Представитель компании предложил услуги по сайту и попросил соединить с маркетологом.",
        }
    )

    assert row["actual_decision"] == "block"
    assert row["passed"] == "yes"


def test_validate_one_case_allows_edtech_context() -> None:
    row = validate_one_case(
        {
            "case_id": "x",
            "expected_decision": "allow",
            "input_text": "Клиент интересовался курсом олимпиадной математики для ребенка 7 класса.",
        }
    )

    assert row["actual_decision"] == "allow"
    assert row["passed"] == "yes"


def test_crm_writeback_relevance_frozen_corpus_passes(tmp_path: Path) -> None:
    summary = validate_crm_writeback_frozen_corpus(
        CrmWritebackCorpusValidationConfig(corpus_jsonl=FIXTURE, out_root=tmp_path / "validation")
    )

    assert summary["passed"] is True
    assert summary["rows"] >= 30
    assert summary["failures"] == 0
    assert "seed_policy" in summary
    assert summary["seed_policy"]["passed_minimum_external_seed_ratio"] is True
    assert summary["rolling_closure"]["can_claim_closed"] is False
