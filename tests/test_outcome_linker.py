from __future__ import annotations

from mango_mvp.insights.outcome_linker import (
    SignalSummary,
    build_outcome_pilot_sample,
    choose_final_outcome,
    classify_amo_rows,
    classify_tallanto_rows,
    config_from_args,
    link_chain_outcome,
    parse_args,
)


def test_tallanto_paid_history_is_strong_positive() -> None:
    signal = classify_tallanto_rows(
        [
            {
                "tallanto_id": "t1",
                "student_type": "8 класс",
                "history_raw": "09.02 оплатили математику, чек на почте. Ребенок занимается в группе.",
            }
        ]
    )

    assert signal.label == "won_paid_or_active"
    assert signal.confidence_tier == "strong"
    assert "tallanto_history_has_paid_terms" in signal.reasons


def test_tallanto_future_payment_is_not_marked_as_paid() -> None:
    signal = classify_tallanto_rows(
        [
            {
                "tallanto_id": "t1",
                "student_type": "8 класс",
                "history_raw": "15.09 жду оплату, обещали оплатить завтра, пока думают.",
            }
        ]
    )

    assert signal.label == "payment_pending"
    assert signal.confidence_tier == "proxy"


def test_outcome_linker_default_off_preserves_legacy_negation_behavior() -> None:
    signal = classify_tallanto_rows(
        [
            {
                "tallanto_id": "t1",
                "student_type": "8 класс",
                "history_raw": "Клиент не оплатил и не записался.",
            }
        ]
    )

    assert signal.label == "won_paid_or_active"
    assert "outcome_model_shadow" not in signal.metadata
    assert "outcome_model_primary" not in signal.metadata


def test_outcome_linker_shadow_reports_allowed_won_to_known_flip_without_changing_label() -> None:
    signal = classify_tallanto_rows(
        [
            {
                "tallanto_id": "t1",
                "student_type": "8 класс",
                "history_raw": "Клиент не оплатил и не записался.",
            }
        ],
        outcome_model_mode="shadow",
    )

    assert signal.label == "won_paid_or_active"
    shadow = signal.metadata["outcome_model_shadow"]
    assert shadow["legacy_label"] == "won_paid_or_active"
    assert shadow["semantic_label"] == "known_student_or_lead"
    assert shadow["primary_allowed"] is True
    assert shadow["label_changed"] is True


def test_outcome_linker_primary_applies_only_allowed_won_to_known_flip() -> None:
    signal = classify_tallanto_rows(
        [
            {
                "tallanto_id": "t1",
                "student_type": "8 класс",
                "history_raw": "Клиент не оплатил и не записался.",
            }
        ],
        outcome_model_mode="primary",
    )

    assert signal.label == "known_student_or_lead"
    assert signal.metadata["outcome_model_primary"]["primary_applied"] is True
    assert signal.metadata["legacy_outcome"]["label"] == "won_paid_or_active"


def test_outcome_linker_primary_blocks_payment_pending_flip() -> None:
    signal = classify_tallanto_rows(
        [
            {
                "tallanto_id": "t1",
                "student_type": "Слушатель",
                "history_raw": "Клиент не оплатил, жду оплату.",
            }
        ],
        outcome_model_mode="primary",
    )

    assert signal.label == "won_paid_or_active"
    primary = signal.metadata["outcome_model_primary"]
    assert primary["semantic_label"] == "payment_pending"
    assert primary["primary_allowed"] is False
    assert primary["primary_applied"] is False
    assert primary["primary_blocked_reason"] == "flip_not_allowlisted"


def test_outcome_linker_cli_accepts_b_primary_mode_without_changing_default() -> None:
    default_config = config_from_args(parse_args([]))
    primary_config = config_from_args(parse_args(["--outcome-model-mode", "primary"]))

    assert default_config.outcome_model_mode == "off"
    assert primary_config.outcome_model_mode == "primary"


def test_outcome_linker_negated_refusal_does_not_negate_paid_after_dash() -> None:
    signal = classify_tallanto_rows(
        [
            {
                "tallanto_id": "t1",
                "student_type": "8 класс",
                "history_raw": "Не отказались - оплатили и придут на занятие.",
            }
        ],
        outcome_model_mode="shadow",
    )

    shadow = signal.metadata["outcome_model_shadow"]
    assert shadow["legacy_label"] == "churn_or_refused_after_activity"
    assert shadow["semantic_label"] == "won_paid_or_active"
    assert shadow["label_changed"] is True


def test_tallanto_future_group_assignment_is_not_marked_as_enrolled() -> None:
    signal = classify_tallanto_rows(
        [
            {
                "tallanto_id": "t1",
                "student_type": "8 класс",
                "history_raw": "25.02 выслала договор, жду инфо по направлению и нужно будет записать в группу",
            }
        ]
    )

    assert signal.label == "in_progress_or_undecided"
    assert "tallanto_history_has_learning_terms" not in signal.reasons


def test_tallanto_latest_refusal_after_activity_is_churn_context() -> None:
    signal = classify_tallanto_rows(
        [
            {
                "tallanto_id": "t1",
                "student_type": "8 класс",
                "history_raw": "09.02 оплатили математику, чек на почте\n15.02 отказ от второго семестра",
            }
        ]
    )

    assert signal.label == "churn_or_refused_after_activity"
    assert signal.latest_signal == "refusal"


def test_amo_reopen_verdict_becomes_reactivation_opportunity() -> None:
    signal = classify_amo_rows(
        [
            {
                "Телефон": "79161492492",
                "ID сделки amoCRM": "1",
                "ID контакта amoCRM": "2",
                "Статус": "Закрыто и не реализовано",
                "AI-вердикт": "reopen_recommended",
                "AI-risk": "high",
                "Основание": "После закрытия был интерес и следующий шаг.",
            }
        ]
    )

    assert signal.label == "reopen_or_follow_up_opportunity"
    assert signal.confidence_tier == "strong"


def test_final_outcome_reports_source_conflict_for_positive_vs_lost() -> None:
    tallanto = SignalSummary("won_paid_or_active", "strong", 0.86, ["paid"])
    amo = SignalSummary("closed_lost_valid", "strong", 0.72, ["closed"])
    calls = SignalSummary("contentful_unknown_outcome", "proxy", 0.3, ["calls"])

    final = choose_final_outcome(tallanto, amo, calls)

    assert final.label == "mixed_outcome_manual_review"
    assert "source_conflict_requires_review" in final.reasons


def test_link_chain_prioritizes_reactivation_use_case() -> None:
    chain = {
        "client_key": "phone:79161492492",
        "phone": "79161492492",
        "years": "2026",
        "contentful_call_count": "3",
        "sales_call_count": "2",
        "existing_client_progress_count": "0",
        "service_call_count": "0",
        "next_step_count": "1",
        "objections_top": "цена: 1",
        "utility_score": "100",
    }
    amo = {
        "79161492492": SignalSummary(
            "reopen_or_follow_up_opportunity",
            "strong",
            0.82,
            ["amo_verdict:reopen_recommended"],
            metadata={"sources": ["AMO"]},
        )
    }

    row = link_chain_outcome(chain, [], {}, amo)

    assert row["sales_action_label"] == "sales_reactivation_candidate"
    assert row["extraction_use_case"] == "reactivation_revenue"
    assert int(row["extraction_priority_score"]) > 100


def test_pilot_sample_keeps_use_case_quotas_and_unique_clients() -> None:
    rows = []
    for idx in range(3):
        rows.append(
            {
                "client_key": f"phone:{idx}",
                "phone": str(idx),
                "extraction_use_case": "reactivation_revenue",
                "extraction_priority_score": str(100 - idx),
            }
        )
    for idx in range(3, 6):
        rows.append(
            {
                "client_key": f"phone:{idx}",
                "phone": str(idx),
                "extraction_use_case": "winner_pattern_for_playbook",
                "extraction_priority_score": str(100 - idx),
            }
        )

    sample = build_outcome_pilot_sample(rows, 4)

    assert len(sample) == 4
    assert len({row["client_key"] for row in sample}) == 4
    assert sample[0]["extraction_use_case"] == "reactivation_revenue"
