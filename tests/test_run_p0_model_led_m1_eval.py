from __future__ import annotations

import json
from pathlib import Path

from mango_mvp.channels.subscription_llm import SubscriptionDraftResult
from scripts import run_p0_model_led_m1_eval as p0_eval


class _Provider:
    def __init__(self) -> None:
        self.calls = 0

    def _direct_path_draft_runner(self, prompt: str) -> SubscriptionDraftResult:
        self.calls += 1
        assert "is_p0" in prompt
        return SubscriptionDraftResult(
            metadata={
                "direct_path_model_p0": {
                    "is_p0": True,
                    "is_p0_present": True,
                    "p0_kind": "complaint",
                }
            }
        )


def test_evaluate_case_uses_one_model_call_and_does_not_return_text_or_pii() -> None:
    provider = _Provider()
    row = p0_eval.evaluate_case(
        {
            "case_index": 1,
            "text": "Жалоба от parent@example.ru, телефон +7 999 123-45-67.",
            "label": "p0",
            "class": "complaint",
            "source": "paraphrase",
            "recent_messages": ("Ранее оплатила, parent@example.ru, +7 999 123-45-67.",),
        },
        provider=provider,  # type: ignore[arg-type]
    )

    serialized = json.dumps(row, ensure_ascii=False)
    assert provider.calls == 1
    assert len(row["case_id"]) == 20
    assert row["model_is_p0"] is True
    assert "parent@example.ru" not in serialized
    assert "999" not in serialized
    assert "text" not in row


def test_load_cases_validates_set_and_summary_uses_fixed_denominator(tmp_path: Path) -> None:
    path = tmp_path / "set.jsonl"
    path.write_text(
        "\n".join(
            (
                json.dumps({"case_id": "traffic_refund_001", "text": "Верните оплату.", "label": "p0", "class": "refund", "source": "traffic_hit"}),
                json.dumps({"text": "Когда занятия?", "label": "benign", "class": "none", "source": "traffic_miss"}),
            )
        ),
        encoding="utf-8",
    )
    cases = p0_eval.load_cases(path)
    assert cases[0]["case_id"] == "traffic_refund_001"
    assert cases[0]["case_id"] != cases[1]["case_id"]
    assert cases[0]["review_status"] == "single_reviewer"
    rows = [
        {**cases[0], "model_is_p0": True, "regex_is_p0": True, "model_field_present": True},
        {**cases[1], "model_is_p0": False, "regex_is_p0": False, "model_field_present": True},
    ]

    summary = p0_eval.summarize(rows, denominator=27_507)

    assert summary["traffic_denominator"] == 27_507
    assert summary["counters"]["model_tp"] == 1
    assert summary["counters"]["model_tn"] == 1


def test_ambiguous_case_is_report_only(tmp_path: Path) -> None:
    path = tmp_path / "set.jsonl"
    path.write_text(
        json.dumps(
            {
                "text": "Можно ли расторгнуть договор?",
                "p0_label": "ambiguous",
                "class": "refund",
                "source": "traffic_hit",
                "review_status": "needs_context",
                "expected_route": "manual_review",
            }
        ),
        encoding="utf-8",
    )
    case = p0_eval.load_cases(path)[0]
    summary = p0_eval.summarize(
        [{**case, "model_is_p0": True, "regex_is_p0": True, "model_field_present": True}],
        denominator=27_507,
    )

    assert case["expected_route"] == "manual_review"
    assert summary["counters"] == {"report_only_ambiguous": 1}


def test_load_cases_stops_before_llm_when_input_contains_person_name(tmp_path: Path) -> None:
    path = tmp_path / "set.jsonl"
    path.write_text(
        json.dumps({"text": "Менеджер Анна Иванова обещала перезвонить.", "p0_label": "benign"}),
        encoding="utf-8",
    )

    try:
        p0_eval.load_cases(path)
    except ValueError as exc:
        assert "PII signals are forbidden" in str(exc)
        assert "person_name" in str(exc)
    else:
        raise AssertionError("PII input must be rejected before any model call")


def test_load_cases_rejects_single_colloquial_names(tmp_path: Path) -> None:
    path = tmp_path / "set.jsonl"
    path.write_text(
        json.dumps({"text": "Лёш, подскажите, когда Юлия перезвонит?", "p0_label": "benign"}),
        encoding="utf-8",
    )

    try:
        p0_eval.load_cases(path)
    except ValueError as exc:
        assert "person_name" in str(exc)
    else:
        raise AssertionError("colloquial names must be rejected before any model call")
