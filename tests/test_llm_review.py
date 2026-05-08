from __future__ import annotations

import json
from pathlib import Path

import pytest

from mango_mvp.insights.llm_review import (
    LLMReviewConfig,
    batch_review_output_json_schema,
    build_codex_cli_command,
    build_batch_review_prompt,
    build_review_prompt,
    codex_cli_env,
    deterministic_review_payload,
    extract_batch_review_payloads,
    flatten_review,
    map_seed_signal,
    map_seed_stage,
    normalize_review_payload,
    normalized_codex_batch_size,
    review_output_json_schema,
    run_pilot_sales_moment_llm_review,
    should_use_codex_batch,
    select_review_items,
)


def _item(idx: int, use_case: str = "reactivation_revenue", signal: str = "next_year_interest") -> dict:
    return {
        "id": f"pilot-{idx:05d}",
        "chain_context": {
            "phone": f"79{idx:09d}",
            "extraction_use_case": use_case,
            "final_outcome_label": "reopen_or_follow_up_opportunity",
            "outcome_confidence_tier": "strong",
        },
        "call_context": {
            "source_filename": f"call-{idx}.mp3",
            "started_at": "2026-04-01 10:00:00",
            "manager_name": "Менеджер",
            "call_type": "sales_call",
            "history_summary": "Клиент спрашивал про следующий год.",
        },
        "deterministic_seed": {
            "customer_question_or_need": "Интересует обучение на следующий год.",
            "customer_signal_label": signal,
            "hidden_sales_stage": "reactivation_after_lost_deal",
            "manager_answer_or_reaction": "Менеджер предложил перезвонить, когда появится расписание.",
            "manager_response_quality_score": 70,
            "ideal_manager_reaction": "Зафиксировать предметы и дату follow-up.",
            "ideal_answer_template": "Когда будет расписание, вернусь с конкретными вариантами.",
        },
        "transcript": "MANAGER: Здравствуйте\nCLIENT: Интересует следующий год",
    }


def _config(*, dry_run: bool = True) -> LLMReviewConfig:
    return LLMReviewConfig(
        project_root=Path("."),
        input_jsonl=Path("input.jsonl"),
        out_root=Path("out"),
        dry_run=dry_run,
    )


def _write_jsonl(path: Path, items: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in items) + "\n", encoding="utf-8")


def test_review_output_json_schema_requires_core_sales_fields() -> None:
    schema = review_output_json_schema()

    required = set(schema["required"])
    rubric = schema["properties"]["rubric_scores"]

    assert "customer_question" in required
    assert "ideal_answer_example" in required
    assert "evidence_quotes" in required
    assert rubric["additionalProperties"] is False
    assert set(rubric["required"]) == {
        "factual_correctness",
        "completeness",
        "persuasiveness",
        "personalization",
        "objection_handling",
        "next_step_clarity",
        "empathy_tone",
        "sales_discipline",
    }


def test_batch_review_schema_and_prompt_require_moment_id() -> None:
    schema = batch_review_output_json_schema()
    item_schema = schema["properties"]["reviews"]["items"]
    prompt = build_batch_review_prompt([_item(1), _item(2)])

    assert schema["required"] == ["reviews"]
    assert "moment_id" in item_schema["required"]
    assert item_schema["properties"]["moment_id"]["type"] == "string"
    assert "pilot-00001" in prompt
    assert "pilot-00002" in prompt


def test_extract_batch_review_payloads_validates_ids() -> None:
    payload_one = deterministic_review_payload(_item(1))
    payload_two = deterministic_review_payload(_item(2))
    payload = {
        "reviews": [
            {"moment_id": "pilot-00001", **payload_one},
            {"moment_id": "pilot-00002", **payload_two},
        ]
    }

    by_id = extract_batch_review_payloads(payload, ["pilot-00001", "pilot-00002"])

    assert set(by_id) == {"pilot-00001", "pilot-00002"}
    assert "moment_id" not in by_id["pilot-00001"]

    with pytest.raises(ValueError, match="missing"):
        extract_batch_review_payloads({"reviews": [{"moment_id": "pilot-00001", **payload_one}]}, ["pilot-00001", "pilot-00002"])
    with pytest.raises(ValueError, match="duplicate"):
        extract_batch_review_payloads(payload, ["pilot-00001", "pilot-00001"])
    with pytest.raises(ValueError, match="unexpected"):
        extract_batch_review_payloads({"reviews": [{"moment_id": "pilot-99999", **payload_one}]}, ["pilot-00001"])


def test_codex_batch_size_is_clamped_and_enabled_only_for_live_codex() -> None:
    config = LLMReviewConfig(
        project_root=Path("."),
        input_jsonl=Path("input.jsonl"),
        out_root=Path("out"),
        provider="codex_cli",
        dry_run=False,
        codex_batch_size=99,
    )

    assert normalized_codex_batch_size(config) == 10
    assert should_use_codex_batch(config) is True
    assert should_use_codex_batch(LLMReviewConfig(Path("."), Path("input.jsonl"), Path("out"), provider="codex_cli", dry_run=True, codex_batch_size=5)) is False
    assert should_use_codex_batch(LLMReviewConfig(Path("."), Path("input.jsonl"), Path("out"), provider="openai", dry_run=False, codex_batch_size=5)) is False


def test_build_codex_cli_command_uses_read_only_schema_and_output() -> None:
    config = LLMReviewConfig(
        project_root=Path("/tmp/project"),
        input_jsonl=Path("input.jsonl"),
        out_root=Path("out"),
        provider="codex_cli",
        model="gpt-5.5",
        reasoning_effort="medium",
        codex_cli_command="codex-bin",
        codex_home=Path("/tmp/codex-home"),
    )

    cmd = build_codex_cli_command(
        config,
        output_path=Path("/tmp/review.json"),
        schema_path=Path("/tmp/schema.json"),
    )

    assert cmd[0] == "codex-bin"
    assert cmd[1] == "exec"
    assert "--ignore-user-config" in cmd
    assert "--ignore-rules" in cmd
    assert cmd[cmd.index("--sandbox") + 1] == "read-only"
    assert cmd[cmd.index("--cd") + 1] == str(Path("/tmp/project").resolve())
    assert cmd[cmd.index("--model") + 1] == "gpt-5.5"
    assert cmd[cmd.index("--output-schema") + 1] == "/tmp/schema.json"
    assert cmd[cmd.index("--output-last-message") + 1] == "/tmp/review.json"
    assert 'model_reasoning_effort="medium"' in cmd
    assert cmd[-1] == "-"

    env = codex_cli_env(config)
    assert env["CODEX_HOME"] == str(Path("/tmp/codex-home").resolve())


def test_select_review_items_stratifies_across_use_case_and_signal() -> None:
    items = [_item(i, "reactivation_revenue", "next_year_interest") for i in range(10)]
    items += [_item(100 + i, "winner_pattern_for_playbook", "payment_service") for i in range(10)]
    items += [_item(200 + i, "loss_pattern_for_objection_playbook", "refusal_or_cooling") for i in range(10)]

    selected = select_review_items(items, limit=6, offset=0, strategy="stratified")

    groups = {
        f"{row['chain_context']['extraction_use_case']}::{row['deterministic_seed']['customer_signal_label']}"
        for row in selected
    }
    assert len(selected) == 6
    assert len(groups) == 3


def test_build_review_prompt_contains_schema_and_input() -> None:
    prompt = build_review_prompt(_item(1))

    assert "return_json_schema" in prompt
    assert "pilot-00001" in prompt
    assert "signal_taxonomy" in prompt


def test_map_seed_signal_and_stage_to_taxonomy() -> None:
    assert map_seed_signal("payment_service") == "payment_or_contract_service"
    assert map_seed_signal("next_year_interest") == "parent_decision_delay"
    assert map_seed_stage("reactivation_after_lost_deal") == "reactivation"
    assert map_seed_stage("", "reactivation_revenue") == "reactivation"


def test_deterministic_review_payload_is_valid_structural_placeholder() -> None:
    payload = deterministic_review_payload(_item(1))

    assert payload["review_schema_version"] == "v1"
    assert payload["customer_signal_type"] == "parent_decision_delay"
    assert payload["hidden_sales_stage"] == "reactivation"
    assert payload["risk_flags"] == ["dry_run_not_llm_review"]
    assert 0 <= payload["overall_quality_score"] <= 100


def test_normalize_review_payload_clamps_scores_and_falls_back_to_seed() -> None:
    item = _item(1)
    payload = {
        "customer_signal_type": "bad-signal",
        "hidden_sales_stage": "bad-stage",
        "manager_answer": "",
        "rubric_scores": {"factual_correctness": 150, "completeness": -10},
        "overall_quality_score": 999,
        "extraction_confidence": 2,
    }

    normalized = normalize_review_payload(payload, item, _config())

    assert normalized["customer_signal_type"] == "parent_decision_delay"
    assert normalized["hidden_sales_stage"] == "reactivation"
    assert normalized["rubric_scores"]["factual_correctness"] == 100
    assert normalized["rubric_scores"]["completeness"] == 0
    assert normalized["overall_quality_score"] == 100
    assert normalized["extraction_confidence"] == 1.0
    assert "перезвонить" in normalized["manager_answer"]


def test_flatten_review_preserves_context_and_rubric_columns() -> None:
    item = _item(1)
    normalized = normalize_review_payload(deterministic_review_payload(item), item, _config())

    row = flatten_review(item, normalized, _config())

    assert row["moment_id"] == "pilot-00001"
    assert row["phone"].startswith("79")
    assert row["rubric_factual_correctness"] != ""
    assert "dry_run_not_llm_review" in row["risk_flags"]


def test_run_live_codex_batch_uses_batch_calls_and_writes_incremental_outputs(tmp_path, monkeypatch) -> None:
    import mango_mvp.insights.llm_review as llm_review

    items = [_item(i) for i in range(1, 4)]
    input_jsonl = tmp_path / "input.jsonl"
    _write_jsonl(input_jsonl, items)
    batch_sizes: list[int] = []

    def fake_batch_provider(_config: LLMReviewConfig, batch_items: list[dict]) -> dict[str, dict]:
        batch_sizes.append(len(batch_items))
        return {item["id"]: deterministic_review_payload(item) for item in batch_items}

    monkeypatch.setattr(llm_review, "call_codex_cli_batch_review_provider", fake_batch_provider)

    summary = run_pilot_sales_moment_llm_review(
        LLMReviewConfig(
            project_root=tmp_path,
            input_jsonl=input_jsonl,
            out_root=tmp_path / "out",
            provider="codex_cli",
            dry_run=False,
            cache_enabled=False,
            force=True,
            codex_batch_size=2,
        )
    )

    assert batch_sizes == [2, 1]
    assert summary["totals"]["reviews_written"] == 3
    assert summary["totals"]["errors"] == 0
    assert summary["totals"]["codex_batch_provider_calls"] == 2
    assert summary["totals"]["codex_batch_fallback_single_calls"] == 0
    assert (tmp_path / "out" / "reviews.jsonl").read_text(encoding="utf-8").count("\n") == 3


def test_run_live_codex_batch_falls_back_to_single_calls(tmp_path, monkeypatch) -> None:
    import mango_mvp.insights.llm_review as llm_review

    items = [_item(i) for i in range(1, 4)]
    input_jsonl = tmp_path / "input.jsonl"
    _write_jsonl(input_jsonl, items)
    single_ids: list[str] = []

    def fake_batch_provider(_config: LLMReviewConfig, _batch_items: list[dict]) -> dict[str, dict]:
        raise ValueError("synthetic batch failure")

    def fake_single_provider(_config: LLMReviewConfig, prompt: str) -> dict:
        for item in items:
            if item["id"] in prompt:
                single_ids.append(item["id"])
                return deterministic_review_payload(item)
        raise AssertionError("prompt did not contain a known item id")

    monkeypatch.setattr(llm_review, "call_codex_cli_batch_review_provider", fake_batch_provider)
    monkeypatch.setattr(llm_review, "call_review_provider", fake_single_provider)

    summary = run_pilot_sales_moment_llm_review(
        LLMReviewConfig(
            project_root=tmp_path,
            input_jsonl=input_jsonl,
            out_root=tmp_path / "out",
            provider="codex_cli",
            dry_run=False,
            cache_enabled=False,
            force=True,
            codex_batch_size=3,
        )
    )

    assert single_ids == ["pilot-00001", "pilot-00002", "pilot-00003"]
    assert summary["totals"]["reviews_written"] == 3
    assert summary["totals"]["errors"] == 0
    assert summary["totals"]["codex_batch_provider_calls"] == 1
    assert summary["totals"]["codex_batch_fallback_single_calls"] == 3
