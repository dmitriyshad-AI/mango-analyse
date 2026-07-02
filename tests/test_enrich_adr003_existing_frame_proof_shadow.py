from __future__ import annotations

import json
from pathlib import Path

from scripts import enrich_adr003_existing_frame_proof_shadow as replay


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _snapshot(tmp_path: Path) -> Path:
    path = tmp_path / "kb_release_v3_snapshot.json"
    path.write_text(
        json.dumps(
            {
                "facts": [
                    {
                        "brand": "unpk",
                        "fact_key": "unpk.olympiad.physics.online_11",
                        "fact_id": "unpk.olympiad.physics.online_11",
                        "fact_type": "program",
                        "product": "олимпиадная физика онлайн",
                        "client_safe_text": (
                            "УНПК: для 11 класса есть онлайн-подготовка по олимпиадной физике."
                        ),
                        "allowed_for_client_answer": True,
                        "forbidden_for_client": False,
                        "internal_only": False,
                        "valid_until": "2026-12-31",
                        "structured_value": {"format": "online", "classes": [11]},
                    }
                ]
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def _frame(**updates: object) -> dict:
    frame = {
        "schema_version": "semantic_frame_v1_2026_07_01",
        "mode": "shadow",
        "intent": "спросить, есть ли олимпиадная физика онлайн для 11 класса",
        "risk_class": "missing_facts",
        "deal_stage": "interest",
        "payment_readiness": "none",
        "requested_product": {
            "brand": "unpk",
            "subject": "physics",
            "grade": "11",
            "format": "online",
            "program_kind": "olympiad",
            "raw_text": "олимпиадная физика онлайн для 11 класса",
        },
        "requested_action": "answer_question",
        "answerability": "manager_only",
        "must_handoff": True,
        "confidence": 0.94,
    }
    frame.update(updates)
    return frame


def _dialog(*, frame: dict | None = None, direct: dict | None = None) -> dict:
    return {
        "dialog_id": "proof_replay_001",
        "brand": "unpk",
        "turns": [
            {
                "turn": 1,
                "client_message": "Есть олимпиадная физика онлайн для 11 класса?",
                "bot_route": "draft_for_manager",
                "bot_text": "Менеджер проверит и подскажет.",
                "bot_message_type": "question",
                "bot_risk_level": "low",
                "bot_safety_flags": ["manager_approval_required", "no_auto_send"],
                "bot_manager_checklist": ["Проверить перед отправкой."],
                "bot_missing_facts": ["актуальная программа"],
                "bot_semantic_frame": frame or {},
                "bot_direct_path": direct
                if direct is not None
                else {
                    "semantic_frame_posthoc_shadow": {"status": "ok"},
                },
            }
        ],
    }


def test_existing_frame_replay_adds_proof_and_reconciliation_without_route_text_change(tmp_path: Path) -> None:
    dialogs, summary = replay.enrich_dialogs([_dialog(frame=_frame())], snapshot_path=_snapshot(tmp_path))

    turn = dialogs[0]["turns"][0]
    proof = turn["bot_semantic_frame_existence_proof_shadow"]
    reconciliation = turn["bot_semantic_frame_proof_reconciliation_shadow"]

    assert turn["bot_route"] == "draft_for_manager"
    assert turn["bot_text"] == "Менеджер проверит и подскажет."
    assert turn["existing_frame_proof_shadow_model_calls_added"] == 0
    assert proof["status"] == "exists"
    assert proof["exact_fact_keys"] == ["unpk.olympiad.physics.online_11"]
    assert reconciliation["status"] == "would_reconcile_to_safe_reference"
    assert reconciliation["source_fact_key"] == "unpk.olympiad.physics.online_11"
    assert summary["route_text_diff_count"] == 0
    assert summary["protected_turn_field_diff_count"] == 0
    assert summary["model_calls_added"] == 0
    assert summary["counts"]["proof_reconciliation_would_reconcile_to_safe_reference"] == 1


def test_existing_frame_replay_skips_missing_frame(tmp_path: Path) -> None:
    dialogs, summary = replay.enrich_dialogs([_dialog(frame={})], snapshot_path=_snapshot(tmp_path))

    turn = dialogs[0]["turns"][0]
    assert turn["existing_frame_proof_shadow_replayed"] is False
    assert turn["existing_frame_proof_shadow_skip_reason"] == "no_bot_semantic_frame"
    assert "bot_semantic_frame_existence_proof_shadow" not in turn
    assert summary["counts"]["turns_missing_frame"] == 1
    assert summary["route_text_diff_count"] == 0


def test_existing_frame_replay_keeps_p0_protected_even_when_frame_says_safe(tmp_path: Path) -> None:
    safe_frame = _frame(risk_class="safe", answerability="answer_self", must_handoff=False)
    direct = {
        "semantic_frame_posthoc_shadow": {"status": "ok"},
        "model_p0": {"is_p0": True, "p0_kind": "refund_claim", "risk_level": "high"},
    }

    dialogs, summary = replay.enrich_dialogs([_dialog(frame=safe_frame, direct=direct)], snapshot_path=_snapshot(tmp_path))

    turn = dialogs[0]["turns"][0]
    shadow = turn["bot_semantic_frame_self_answer_shadow"]
    assert shadow["status"] == "blocked"
    assert shadow["reason"] == "protected_p0"
    assert shadow["route_after_if_active"] == "draft_for_manager"
    assert summary["counts"]["self_answer_shadow_reason:protected_p0"] == 1
    assert summary["route_text_diff_count"] == 0


def test_existing_frame_replay_cli_writes_enriched_artifacts(tmp_path: Path) -> None:
    transcripts = tmp_path / "input.jsonl"
    out_dir = tmp_path / "out"
    _write_jsonl(transcripts, [_dialog(frame=_frame())])

    rc = replay.main(
        [
            "--transcripts",
            str(transcripts),
            "--kb-snapshot",
            str(_snapshot(tmp_path)),
            "--out-dir",
            str(out_dir),
            "--input-source-rev",
            "36ea110",
        ]
    )

    assert rc == 0
    summary = json.loads((out_dir / "existing_frame_proof_shadow_summary.json").read_text(encoding="utf-8"))
    enriched = json.loads((out_dir / "dynamic_dialog_transcripts.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert summary["inputs"]["transcripts_sha256"]
    assert summary["input_source_rev"] == "36ea110"
    assert summary["model_calls_added"] == 0
    assert enriched["turns"][0]["bot_semantic_frame_proof_reconciliation_shadow"]["status"] == (
        "would_reconcile_to_safe_reference"
    )
    assert (out_dir / "existing_frame_proof_shadow_summary.md").exists()


def test_existing_frame_replay_cli_refuses_stable_runtime_outdir(tmp_path: Path) -> None:
    transcripts = tmp_path / "input.jsonl"
    _write_jsonl(transcripts, [_dialog(frame=_frame())])

    try:
        replay.main(
            [
                "--transcripts",
                str(transcripts),
                "--kb-snapshot",
                str(_snapshot(tmp_path)),
                "--out-dir",
                str(tmp_path / "stable_runtime" / "replay"),
            ]
        )
    except SystemExit as exc:
        assert "stable_runtime" in str(exc)
    else:
        raise AssertionError("stable_runtime out-dir must be rejected")
