from __future__ import annotations

import json
from pathlib import Path

from scripts import report_adr003_overhandoff_levers as report


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _gold(dialog_id: str, *, must_handoff: bool = False, notes: str = "") -> dict:
    return {
        "dialog_id": dialog_id,
        "turn": 1,
        "expected": {
            "must_handoff": must_handoff,
            "risk_class": "safe" if not must_handoff else "payment_dispute",
            "answerability": "answer_self" if not must_handoff else "manager_only",
            "requested_action": "answer_question",
        },
        "review_label": "frame_correct",
        "notes": notes,
    }


def _turn(
    dialog_id: str,
    *,
    route: str = "manager_only",
    client_message: str = "Аккаунт активировали, всё работает",
    bot_text: str = "Спасибо, зафиксировали.",
    message_type: str = "context_update",
    frame: dict | None = None,
    contract: dict | None = None,
    safety_flags: list[str] | None = None,
    missing_facts: list[str] | None = None,
    actual_p0: bool = False,
    freshness: dict | None = None,
    self_class: str = "context_ack",
) -> dict:
    semantic_frame = {
        "risk_class": "safe",
        "answerability": "answer_self",
        "requested_action": "answer_question",
        "deal_stage": "qualification",
        "payment_readiness": "none",
        "must_handoff": False,
        "confidence": 0.94,
    }
    semantic_frame.update(frame or {})
    answer_contract = {
        "route": route,
        "route_reason": "help_then_one_question",
        "primary_intent": "general_consultation",
        "answer_policy": "help_then_one_question",
        "direct_question": "",
        "p0_required": False,
    }
    answer_contract.update(contract or {})
    guards = {
        "actual_p0": actual_p0,
        "has_missing_facts": bool(missing_facts),
        "blocking_flags": [],
        "freshness": freshness
        if freshness is not None
        else {
            "ok": False,
            "exact_fact_count": 0,
            "fresh_client_safe_count": 0,
            "all_exact_facts_fresh_client_safe": False,
        },
    }
    return {
        "dialog_id": dialog_id,
        "brand": "foton",
        "turns": [
            {
                "turn": 1,
                "client_message": client_message,
                "bot_text": bot_text,
                "bot_route": route,
                "bot_message_type": message_type,
                "bot_safety_flags": safety_flags or ["manager_approval_required", "no_auto_send"],
                "bot_answer_contract": answer_contract,
                "bot_semantic_frame": semantic_frame,
                "bot_semantic_frame_self_answer_shadow": {
                    "status": "blocked",
                    "reason": "route_not_draft_for_manager" if route == "manager_only" else "low_confidence",
                    "self_class": self_class,
                    "guards": guards,
                },
                "bot_missing_facts": missing_facts or [],
            }
        ],
    }


def _build(tmp_path: Path, dialogs: list[dict], gold_rows: list[dict]) -> dict:
    transcripts = tmp_path / "transcripts.jsonl"
    gold = tmp_path / "gold.jsonl"
    _write_jsonl(transcripts, dialogs)
    _write_jsonl(gold, gold_rows)
    return report.build_report(transcripts=transcripts, gold=gold)


def test_manager_only_context_ack_is_diagnostic_candidate(tmp_path: Path) -> None:
    result = _build(tmp_path, [_turn("d1")], [_gold("d1")])

    assert result["totals"]["harmless_context_ack_status_candidates"] == 1
    candidate = result["groups"]["harmless_context_ack_status_candidate"]["examples"][0]
    assert candidate["candidate_status"] == "would_need_manager_only_policy_decision"
    assert candidate["blocked_reasons"] == []


def test_draft_for_manager_context_ack_is_future_route_only_candidate(tmp_path: Path) -> None:
    result = _build(tmp_path, [_turn("d1", route="draft_for_manager")], [_gold("d1")])

    candidate = result["groups"]["harmless_context_ack_status_candidate"]["examples"][0]
    assert candidate["candidate_status"] == "would_allow_self_context_ack"
    assert result["totals"]["draft_candidates_for_future_active"] == 1


def test_access_problem_is_blocked_by_runtime_p0_guard(tmp_path: Path) -> None:
    result = _build(
        tmp_path,
        [
            _turn(
                "d1",
                client_message="Аккаунт активировали, но доступ не работает",
                safety_flags=["manager_approval_required", "p0_model_led"],
                actual_p0=True,
            )
        ],
        [_gold("d1")],
    )

    blocked = result["groups"]["p0_or_money_or_operational_blocked"]["examples"][0]
    assert "runtime_actual_p0" in blocked["blocked_reasons"]


def test_manager_action_is_blocked(tmp_path: Path) -> None:
    result = _build(
        tmp_path,
        [
            _turn(
                "d1",
                client_message="Можно записаться?",
                frame={"requested_action": "enroll", "deal_stage": "closing"},
                contract={"direct_question": "Можно записаться?"},
            )
        ],
        [_gold("d1")],
    )

    blocked = result["groups"]["p0_or_money_or_operational_blocked"]["examples"][0]
    assert "requested_action_not_safe_ack" in blocked["blocked_reasons"]
    assert "money_or_operational_signal" in blocked["blocked_reasons"]
    assert "direct_question_present" in blocked["blocked_reasons"]


def test_missing_facts_are_blocked_as_safe_reference_without_exact_facts(tmp_path: Path) -> None:
    result = _build(
        tmp_path,
        [
            _turn(
                "d1",
                client_message="Расскажите про обе",
                missing_facts=["актуальные цены", "форматы"],
            )
        ],
        [_gold("d1")],
    )

    blocked = result["groups"]["safe_reference_without_exact_facts"]["examples"][0]
    assert "missing_facts" in blocked["blocked_reasons"]


def test_existence_format_question_is_blocked_until_fact_verification(tmp_path: Path) -> None:
    result = _build(
        tmp_path,
        [
            _turn(
                "d1",
                client_message="Есть онлайн-курс для 9 класса?",
                frame={"requested_action": "check_availability"},
                self_class="safe_reference",
            )
        ],
        [_gold("d1", notes="safe reference: course existence/format without live seats")],
    )

    blocked = result["groups"]["existence_format_needs_fact_verification_blocked"]["examples"][0]
    assert "requires_existence_fact_verification" in blocked["blocked_reasons"]
    assert blocked["requires_existence_fact_verification"] is True
    assert result["totals"]["existence_format_needs_fact_verification_blocked"] == 1


def test_frame_too_cautious_tracks_existence_even_when_current_route_is_self(tmp_path: Path) -> None:
    result = _build(
        tmp_path,
        [
            _turn(
                "d1",
                route="bot_answer_self_for_pilot",
                client_message="Есть онлайн-курс для 9 класса?",
                frame={
                    "must_handoff": True,
                    "risk_class": "manager_action",
                    "answerability": "manager_only",
                    "requested_action": "check_availability",
                    "confidence": 0.93,
                },
                self_class="safe_reference",
            )
        ],
        [_gold("d1", notes="safe reference: course existence/format without live seats")],
    )

    assert result["totals"]["safe_already_self"] == 1
    assert result["frame_too_cautious"]["count"] == 1
    assert result["frame_too_cautious"]["by_requested_action"] == {"check_availability": 1}
    assert result["frame_too_cautious"]["existence_format_count"] == 1


def test_danger_adjacent_safe_label_is_separated_from_clean_candidates(tmp_path: Path) -> None:
    result = _build(
        tmp_path,
        [
            _turn(
                "p0_paid_transfer_case",
                frame={"requested_action": "check_availability"},
                self_class="safe_reference",
            )
        ],
        [_gold("p0_paid_transfer_case", notes="safe reference: camp existence without live seats")],
    )

    blocked = result["groups"]["danger_adjacent_blocked"]["examples"][0]
    assert "danger_adjacent_dialog" in blocked["blocked_reasons"]


def test_partial_freshness_is_blocked(tmp_path: Path) -> None:
    result = _build(
        tmp_path,
        [
            _turn(
                "d1",
                freshness={
                    "ok": True,
                    "exact_fact_count": 2,
                    "fresh_client_safe_count": 1,
                    "all_exact_facts_fresh_client_safe": False,
                },
            )
        ],
        [_gold("d1")],
    )

    blocked = result["groups"]["safe_reference_without_exact_facts"]["examples"][0]
    assert "facts_not_fresh_client_safe" in blocked["blocked_reasons"]


def test_unknown_brand_is_blocked(tmp_path: Path) -> None:
    dialog = _turn("d1")
    dialog["brand"] = "unknown"
    result = _build(tmp_path, [dialog], [_gold("d1")])

    blocked = result["groups"]["p0_or_money_or_operational_blocked"]["examples"][0]
    assert "brand_scope_unclear_or_mixed" in blocked["blocked_reasons"]


def test_cross_brand_frame_is_blocked(tmp_path: Path) -> None:
    result = _build(
        tmp_path,
        [
            _turn(
                "d1",
                frame={"requested_product": {"brand": "unpk"}},
            )
        ],
        [_gold("d1")],
    )

    blocked = result["groups"]["p0_or_money_or_operational_blocked"]["examples"][0]
    assert "brand_scope_unclear_or_mixed" in blocked["blocked_reasons"]


def test_unsafe_output_text_without_verified_facts_is_blocked(tmp_path: Path) -> None:
    result = _build(
        tmp_path,
        [
            _turn(
                "d1",
                bot_text="Менеджер свяжется и забронирует место.",
            )
        ],
        [_gold("d1")],
    )

    blocked = result["groups"]["p0_or_money_or_operational_blocked"]["examples"][0]
    assert "unsafe_output_text" in blocked["blocked_reasons"]


def test_low_confidence_is_blocked(tmp_path: Path) -> None:
    result = _build(
        tmp_path,
        [_turn("d1", frame={"confidence": 0.86})],
        [_gold("d1")],
    )

    blocked = result["groups"]["low_confidence_or_missing_facts_blocked"]["examples"][0]
    assert "low_confidence" in blocked["blocked_reasons"]


def test_report_redacts_pii_in_examples(tmp_path: Path) -> None:
    result = _build(
        tmp_path,
        [
            _turn(
                "d1",
                client_message="Меня зовут Анна, телефон +7 900 111-22-33, email anna@example.com, id 1234567",
            )
        ],
        [_gold("d1")],
    )

    candidate = result["groups"]["harmless_context_ack_status_candidate"]["examples"][0]
    assert "+7 900" not in candidate["client_excerpt"]
    assert "anna@example.com" not in candidate["client_excerpt"]
    assert "1234567" not in candidate["client_excerpt"]
    assert "[phone]" in candidate["client_excerpt"]
    assert "[email]" in candidate["client_excerpt"]
    assert "[id]" in candidate["client_excerpt"]
