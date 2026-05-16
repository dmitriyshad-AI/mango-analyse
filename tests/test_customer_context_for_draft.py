from __future__ import annotations

from mango_mvp.channels.customer_context_for_draft import (
    CHILD_CLARIFICATION_TEXT,
    build_customer_context_for_draft,
    customer_context_for_draft_safety_contract,
)


def test_family_phone_requires_child_clarification() -> None:
    context = build_customer_context_for_draft(
        "+7 900 000-00-01",
        incoming_text="Когда можно прийти на пробное?",
        amo_context={"family_phone": True, "lead_status": "existing"},
        tallanto_context={
            "match_status": "multiple_tallanto_matches",
            "students": [
                {"student_id": "s-1", "name": "Маша"},
                {"student_id": "s-2", "name": "Петя"},
            ],
        },
    )

    assert context["requires_child_clarification"] is True
    assert context["draft_requirements"]["must_ask_child_clarification"] is True
    assert context["draft_requirements"]["child_clarification_question"] == CHILD_CLARIFICATION_TEXT
    assert CHILD_CLARIFICATION_TEXT in context["safe_draft_text"]
    assert CHILD_CLARIFICATION_TEXT in context["manager_checklist"]
    assert "child_clarification_required" in context["safety_flags"]
    assert context["sources"]["tallanto"]["students_count"] == 2
    assert context["safety"]["write_amo"] is False
    assert context["safety"]["write_tallanto"] is False


def test_no_tallanto_for_new_lead_is_warning_not_hard_block() -> None:
    context = build_customer_context_for_draft(
        "+7 900 000-00-02",
        incoming_text="Здравствуйте, хочу узнать про подготовку",
        amo_context={"is_new_lead": True, "lead_status": "new_lead"},
        tallanto_context={"match_status": "no_match"},
    )

    assert context["hard_block"] is False
    assert context["blocked"] is False
    assert context["route"] == "draft_for_manager"
    assert context["requires_manager_review"] is True
    assert "tallanto_not_found_for_new_lead" in context["warnings"]
    assert context["sources"]["tallanto"]["present"] is False
    assert context["sources"]["tallanto"]["missing_is_hard_block"] is False


def test_payment_context_requires_manager_review() -> None:
    context = build_customer_context_for_draft(
        "+7 900 000-00-03",
        incoming_text="Как оплатить обучение и какие документы нужны для налоговой справки?",
        amo_context={"lead_status": "existing"},
        tallanto_context={"match_status": "exact_phone_single", "student_id": "s-3"},
    )

    assert context["payment_or_documents_review_required"] is True
    assert context["requires_manager_review"] is True
    assert context["route"] in {"draft_for_manager", "manager_only"}
    assert "payment_or_documents_manager_review_required" in context["safety_flags"]
    assert context["draft_requirements"]["payment_or_documents_require_manager_review"] is True
    assert any("Проверить оплату" in item for item in context["manager_checklist"])
    assert "после ручной проверки" in context["safe_draft_text"]


def test_crm_recommendation_is_not_live_write() -> None:
    context = build_customer_context_for_draft(
        "+7 900 000-00-04",
        incoming_text="Можно записаться?",
        amo_context={"lead_status": "new_lead"},
        tallanto_context={"match_status": "no_match"},
        crm_recommendations=[
            {
                "target": "AMO",
                "action": "update_field",
                "text": "Клиент просит запись на пробное занятие.",
                "requires_manager_approval": False,
                "live_write_enabled": True,
            }
        ],
    )

    assert context["crm_recommendations"]
    for recommendation in context["crm_recommendations"]:
        assert recommendation["action"] == "text_suggestion"
        assert recommendation["text"]
        assert recommendation["requires_manager_approval"] is True
        assert recommendation["live_write_enabled"] is False

    safety = customer_context_for_draft_safety_contract()
    assert safety["write_crm"] is False
    assert safety["write_amo"] is False
    assert safety["write_tallanto"] is False
    assert safety["send_messenger"] is False
