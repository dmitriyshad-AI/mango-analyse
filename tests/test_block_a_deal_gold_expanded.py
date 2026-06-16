from pathlib import Path

from scripts.run_block_a_deal_gold_expanded import (
    TARGET_LOSS_REASONS,
    build_summary,
    classify_error_type,
    infer_brand,
    manual_gold_label,
    select_additional_cases,
)


def _record(*, loss_reason: str, history: str = "", brand: str = "foton", confidence: float = 0.9) -> dict:
    return {
        "case_id": "case_1",
        "heuristic_analysis": {
            "brand": brand,
            "loss_reason_summary": loss_reason,
            "history_summary": history,
            "latest_call_summary": "",
            "recommended_next_step": "",
        },
        "llm_analysis": {
            "deal_summary": history,
            "close_reason_summary": loss_reason,
            "close_verdict": "closed_valid",
            "premature_close_risk": "no_risk",
            "confidence": confidence,
        },
    }


def test_manual_gold_marks_active_client_as_valid_close() -> None:
    gold = manual_gold_label(_record(loss_reason="Действующий клиент"))

    assert gold["gold_verdict"] == "closed_valid"
    assert gold["gold_risk"] == "no_risk"
    assert gold["gold_next_step_class"] == "no_action"
    assert gold["gold_reason"] == "manual_policy_active_client_loss_reason"


def test_manual_gold_requires_follow_up_for_no_answer_without_decline_context() -> None:
    gold = manual_gold_label(_record(loss_reason="Недозвон", history="Менеджер звонил, ответа не было."))

    assert gold["gold_verdict"] == "follow_up_needed"
    assert gold["gold_risk"] == "medium"
    assert gold["gold_next_step_class"] == "follow_up_check"


def test_manual_gold_accepts_duplicate_or_non_sales_close() -> None:
    gold = manual_gold_label(_record(loss_reason="Дубль заявки"))

    assert gold["gold_verdict"] == "closed_valid"
    assert gold["gold_risk"] == "no_risk"
    assert gold["gold_next_step_class"] == "no_action"


def test_infer_brand_supports_both_foton_and_unpk_markers() -> None:
    contact_by_id = {
        "10": {"contact_name": "Родитель", "custom_field_values_json": '{"utm_source": "cdpofoton"}'},
        "11": {"contact_name": "Родитель", "custom_field_values_json": '{"source": "УНПК МФТИ"}'},
    }

    assert infer_brand({"linked_contact_ids": "10", "lead_name": ""}, contact_by_id=contact_by_id) == "foton"
    assert infer_brand({"linked_contact_ids": "11", "lead_name": ""}, contact_by_id=contact_by_id) == "unpk"


def test_select_additional_cases_picks_one_case_per_brand_and_loss_reason() -> None:
    contacts = {
        "1": {"contact_id": "1", "phones": "+7 900 000 00 01", "custom_field_values_json": '{"brand": "Фотон"}'},
        "2": {"contact_id": "2", "phones": "+7 900 000 00 02", "custom_field_values_json": '{"brand": "УНПК"}'},
    }
    deals = []
    for brand_index, (brand_name, contact_id) in enumerate([("Фотон", "1"), ("УНПК", "2")], start=1):
        for loss_index, loss_reason in enumerate(TARGET_LOSS_REASONS, start=1):
            deals.append(
                {
                    "lead_id": f"{brand_index}{loss_index}",
                    "status_id": "143",
                    "loss_reason": loss_reason,
                    "lead_name": brand_name,
                    "linked_contact_ids": contact_id,
                    "custom_field_values_json": "{}",
                }
            )

    selected = select_additional_cases(
        deals=deals,
        contact_by_id=contacts,
        old_ids=set(),
        normalize_phone=lambda value: "".join(ch for ch in value if ch.isdigit()),
    )

    assert len(selected) == 2
    assert {row["brand"] for row in selected} == {"foton", "unpk"}


def test_summary_keeps_model_shadow_when_it_has_confident_wrong_rows() -> None:
    trace_rows = [
        {
            "id": "a",
            "brand": "foton",
            "rule_matches_gold": "Да",
            "model_matches_gold": "Нет",
            "confidence": "0.900000",
            "error_type": "model_break",
        },
        {
            "id": "b",
            "brand": "unpk",
            "rule_matches_gold": "Да",
            "model_matches_gold": "Да",
            "confidence": "0.950000",
            "error_type": "both_correct",
        },
    ]

    summary = build_summary(
        records=[{}, {}],
        trace_rows=trace_rows,
        old_count=1,
        new_count=1,
        out_dir=Path("audits/_inbox/example"),
        snapshot_root=Path("stable_runtime/example"),
    )

    assert summary["decision"] == "keep_shadow"
    assert summary["high_confidence_wrong_count"] == 1
    assert summary["high_confidence_wrong_ids"] == ["a"]
    assert summary["safety"]["writes_amo"] is False
    assert summary["safety"]["writes_tallanto"] is False


def test_classify_error_type_names_model_regression() -> None:
    assert classify_error_type(rule_ok=True, model_ok=False) == "model_break"
    assert classify_error_type(rule_ok=False, model_ok=True) == "model_fix"
    assert classify_error_type(rule_ok=True, model_ok=True) == "both_correct"
    assert classify_error_type(rule_ok=False, model_ok=False) == "both_wrong"
