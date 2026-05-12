from __future__ import annotations

import csv
import json
import sqlite3
from dataclasses import replace
from pathlib import Path

from mango_mvp.db import build_session_factory, init_db
from mango_mvp.models import CallRecord
from mango_mvp.quality.transcript_quality_backfill import (
    TranscriptQualityBackfillConfig,
    run_transcript_quality_backfill,
)
from tests.test_dialogue_format import make_settings


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _analysis(call_type: str) -> str:
    return json.dumps(
        {
            "analysis_schema_version": "v2",
            "history_summary": "old",
            "quality_flags": {"call_type": call_type, "mode": "stereo"},
            "tags": [call_type],
        },
        ensure_ascii=False,
    )


def _row(db_path: Path, source_filename: str) -> dict[str, object]:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        row = con.execute("select * from call_records where source_filename=?", (source_filename,)).fetchone()
        assert row is not None
        return dict(row)
    finally:
        con.close()


def _candidate(
    *,
    row_id: int,
    source_filename: str,
    current_call_type: str,
    current_contentful: bool,
    review_decision: str = "safe_auto_apply_candidate",
    review_hash: str,
    reason_codes: str = "system_no_dialogue_phrase|no_live_marker|asr_artifact_marker",
    recommended_contact_subtype: str = "",
) -> dict[str, object]:
    return {
        "id": row_id,
        "source_filename": source_filename,
        "current_call_type": current_call_type,
        "current_contentful": str(current_contentful),
        "analysis_status": "pending",
        "resolve_status": "manual",
        "guardrail_label": "non_conversation_high_confidence",
        "guardrail_score": -9,
        "guardrail_reason_codes": reason_codes,
        "recommended_contact_subtype": recommended_contact_subtype,
        "should_force_non_conversation": "True",
        "review_decision": review_decision,
        "review_hash": review_hash,
    }


def _llm_consensus_candidate(
    *,
    row_id: int,
    source_filename: str,
    confidence: str = "0.96",
) -> dict[str, object]:
    row = _candidate(
        row_id=row_id,
        source_filename=source_filename,
        current_call_type="unknown",
        current_contentful=False,
        review_hash="hash-llm-consensus",
        reason_codes="client_turn_80_plus|no_live_marker|short_no_live_transcript",
        recommended_contact_subtype="probable_no_live",
    )
    row.update(
        {
            "guardrail_label": "manual_review_probable_no_live",
            "guardrail_score": 1,
            "should_force_non_conversation": "False",
            "consensus_route": "auto_apply_force_non_conversation",
            "consensus_reason": "advanced_high_confidence_force",
            "final_source": "advanced",
            "final_decision": "force_non_conversation",
            "advanced_confidence": confidence,
            "mini_confidence": "",
        }
    )
    return row


def _hard_gate_consensus_candidate(
    *,
    row_id: int,
    source_filename: str,
    consensus_queue: str = "consensus_auto_apply",
    policy_queue: str = "gpt_auto_apply",
    gpt_decision: str = "safe_apply",
    claude_decision: str = "safe_apply",
    review_decision: str = "hard_gate_gpt_auto_apply",
) -> dict[str, object]:
    row = _candidate(
        row_id=row_id,
        source_filename=source_filename,
        current_call_type="sales_call",
        current_contentful=True,
        review_decision=review_decision,
        review_hash=f"hash-{source_filename}",
        reason_codes="system_no_dialogue_phrase|no_live_marker",
        recommended_contact_subtype="no_live_or_voicemail",
    )
    row.update(
        {
            "consensus_queue": consensus_queue,
            "policy_queue": policy_queue,
            "gpt_decision": gpt_decision,
            "claude_decision": claude_decision,
            "policy_auto_apply_allowed": str(policy_queue == "gpt_auto_apply" and gpt_decision == "safe_apply"),
        }
    )
    return row


def test_transcript_quality_backfill_dry_run_apply_and_idempotency(tmp_path: Path) -> None:
    db_path = tmp_path / "calls.db"
    settings = replace(make_settings(), database_url=f"sqlite:///{db_path}")
    init_db(settings)
    session_factory = build_session_factory(settings)

    with session_factory() as session:
        safe_existing = CallRecord(
            source_file=str(tmp_path / "safe_existing.mp3"),
            source_filename="safe_existing.mp3",
            transcription_status="done",
            resolve_status="skipped",
            analysis_status="done",
            transcript_text="MANAGER: Продолжение следует... CLIENT: Номер недоступен. Оставьте сообщение.",
            analysis_json=_analysis("non_conversation"),
        )
        safe_missing = CallRecord(
            source_file=str(tmp_path / "safe_missing.mp3"),
            source_filename="safe_missing.mp3",
            transcription_status="done",
            resolve_status="manual",
            analysis_status="pending",
            transcript_text="MANAGER: Продолжение следует... CLIENT: Абонент сейчас не может ответить.",
            analysis_json=None,
        )
        unsafe_contentful = CallRecord(
            source_file=str(tmp_path / "unsafe_contentful.mp3"),
            source_filename="unsafe_contentful.mp3",
            transcription_status="done",
            resolve_status="done",
            analysis_status="done",
            transcript_text="MANAGER: Добрый день. CLIENT: Мне нужен курс.",
            analysis_json=_analysis("service_call"),
        )
        session.add_all([safe_existing, safe_missing, unsafe_contentful])
        session.commit()
        safe_existing_id = int(safe_existing.id)
        safe_missing_id = int(safe_missing.id)
        unsafe_contentful_id = int(unsafe_contentful.id)

    candidates_csv = tmp_path / "candidates.csv"
    _write_csv(
        candidates_csv,
        [
            _candidate(
                row_id=safe_existing_id,
                source_filename="safe_existing.mp3",
                current_call_type="non_conversation",
                current_contentful=False,
                review_hash="hash-safe-existing",
            ),
            _candidate(
                row_id=safe_missing_id,
                source_filename="safe_missing.mp3",
                current_call_type="unknown",
                current_contentful=False,
                review_hash="hash-safe-missing",
            ),
            _candidate(
                row_id=unsafe_contentful_id,
                source_filename="unsafe_contentful.mp3",
                current_call_type="service_call",
                current_contentful=True,
                review_decision="human_review_required_contentful",
                review_hash="hash-unsafe-contentful",
            ),
        ],
    )

    dry_summary = run_transcript_quality_backfill(
        TranscriptQualityBackfillConfig(
            database_url=settings.database_url,
            candidates_csv=candidates_csv,
            out_root=tmp_path / "dry_run",
            mode="dry-run",
        )
    )

    assert dry_summary["planned_updates"] == 2
    assert dry_summary["applied_updates"] == 0
    assert dry_summary["blocked_rows"] == 1
    assert _row(db_path, "safe_missing.mp3")["analysis_json"] is None

    apply_summary = run_transcript_quality_backfill(
        TranscriptQualityBackfillConfig(
            database_url=settings.database_url,
            candidates_csv=candidates_csv,
            out_root=tmp_path / "apply",
            mode="apply",
        )
    )

    assert apply_summary["planned_updates"] == 2
    assert apply_summary["applied_updates"] == 2
    assert apply_summary["blocked_rows"] == 1
    assert apply_summary["backup_path"]
    assert Path(str(apply_summary["backup_path"])).exists()

    updated_missing = _row(db_path, "safe_missing.mp3")
    payload = json.loads(str(updated_missing["analysis_json"]))
    assert updated_missing["analysis_status"] == "done"
    assert updated_missing["resolve_status"] == "skipped"
    assert payload["quality_flags"]["call_type"] == "non_conversation"
    assert payload["quality_flags"]["transcript_quality_backfill"]["version"] == "safe_non_contentful_v1"
    assert payload["quality_flags"]["transcript_quality_backfill"]["review_hash"] == "hash-safe-missing"
    assert payload["follow_up_score"] == 0
    assert payload["next_step"] is None

    unchanged = json.loads(str(_row(db_path, "unsafe_contentful.mp3")["analysis_json"]))
    assert unchanged["quality_flags"]["call_type"] == "service_call"

    repeat_summary = run_transcript_quality_backfill(
        TranscriptQualityBackfillConfig(
            database_url=settings.database_url,
            candidates_csv=candidates_csv,
            out_root=tmp_path / "repeat",
            mode="apply",
            create_backup=False,
        )
    )

    assert repeat_summary["planned_updates"] == 0
    assert repeat_summary["applied_updates"] == 0
    assert repeat_summary["already_applied"] == 2
    assert repeat_summary["blocked_rows"] == 1


def test_transcript_quality_backfill_allows_safe_outbound_voicemail_from_contentful_call(tmp_path: Path) -> None:
    db_path = tmp_path / "calls.db"
    settings = replace(make_settings(), database_url=f"sqlite:///{db_path}")
    init_db(settings)
    session_factory = build_session_factory(settings)

    with session_factory() as session:
        call = CallRecord(
            source_file=str(tmp_path / "outbound_voicemail.mp3"),
            source_filename="outbound_voicemail.mp3",
            transcription_status="done",
            resolve_status="done",
            analysis_status="done",
            transcript_text=(
                "MANAGER: Добрый день, это Фотон. Оставляю информацию по курсу ЕГЭ по математике.\n"
                "CLIENT: Абонент сейчас не может ответить. Оставьте сообщение после звукового сигнала."
            ),
            analysis_json=_analysis("sales_call"),
        )
        session.add(call)
        session.commit()
        call_id = int(call.id)

    candidates_csv = tmp_path / "candidates.csv"
    _write_csv(
        candidates_csv,
        [
            _candidate(
                row_id=call_id,
                source_filename="outbound_voicemail.mp3",
                current_call_type="sales_call",
                current_contentful=True,
                review_hash="hash-outbound-voicemail",
                reason_codes="system_no_dialogue_phrase|no_live_marker|outbound_voicemail",
                recommended_contact_subtype="outbound_voicemail",
            )
        ],
    )

    summary = run_transcript_quality_backfill(
        TranscriptQualityBackfillConfig(
            database_url=settings.database_url,
            candidates_csv=candidates_csv,
            out_root=tmp_path / "apply",
            mode="apply",
        )
    )

    assert summary["planned_updates"] == 1
    assert summary["blocked_rows"] == 0
    updated = _row(db_path, "outbound_voicemail.mp3")
    payload = json.loads(str(updated["analysis_json"]))
    assert payload["quality_flags"]["call_type"] == "non_conversation"
    assert payload["quality_flags"]["transcript_quality_recommended_contact_subtype"] == "outbound_voicemail"
    assert "пытался связаться" in payload["history_summary"]
    assert "оставил сообщение" in payload["summary"]


def test_transcript_quality_backfill_allows_safe_llm_consensus_candidate(tmp_path: Path) -> None:
    db_path = tmp_path / "calls.db"
    settings = replace(make_settings(), database_url=f"sqlite:///{db_path}")
    init_db(settings)
    session_factory = build_session_factory(settings)

    with session_factory() as session:
        call = CallRecord(
            source_file=str(tmp_path / "llm_consensus.mp3"),
            source_filename="llm_consensus.mp3",
            transcription_status="done",
            resolve_status="manual",
            analysis_status="pending",
            transcript_text="CLIENT: Абонент не берет трубку. Попробуйте перезвонить позднее.",
            analysis_json=None,
        )
        session.add(call)
        session.commit()
        call_id = int(call.id)

    candidates_csv = tmp_path / "candidates.csv"
    _write_csv(
        candidates_csv,
        [_llm_consensus_candidate(row_id=call_id, source_filename="llm_consensus.mp3")],
    )

    summary = run_transcript_quality_backfill(
        TranscriptQualityBackfillConfig(
            database_url=settings.database_url,
            candidates_csv=candidates_csv,
            out_root=tmp_path / "apply",
            mode="apply",
        )
    )

    assert summary["planned_updates"] == 1
    assert summary["blocked_rows"] == 0
    payload = json.loads(str(_row(db_path, "llm_consensus.mp3")["analysis_json"]))
    quality = payload["quality_flags"]
    assert quality["call_type"] == "non_conversation"
    assert quality["transcript_quality_label"] == "llm_consensus_force_non_conversation"
    assert quality["transcript_quality_backfill"]["source_consensus_route"] == "auto_apply_force_non_conversation"
    assert quality["transcript_quality_backfill"]["source_final_source"] == "advanced"
    assert quality["transcript_quality_backfill"]["source_advanced_confidence"] == "0.96"


def test_transcript_quality_backfill_blocks_low_confidence_llm_consensus_candidate(tmp_path: Path) -> None:
    db_path = tmp_path / "calls.db"
    settings = replace(make_settings(), database_url=f"sqlite:///{db_path}")
    init_db(settings)
    session_factory = build_session_factory(settings)

    with session_factory() as session:
        call = CallRecord(
            source_file=str(tmp_path / "low_confidence.mp3"),
            source_filename="low_confidence.mp3",
            transcription_status="done",
            resolve_status="manual",
            analysis_status="pending",
            transcript_text="CLIENT: Абонент не берет трубку.",
        )
        session.add(call)
        session.commit()
        call_id = int(call.id)

    candidates_csv = tmp_path / "candidates.csv"
    _write_csv(
        candidates_csv,
        [_llm_consensus_candidate(row_id=call_id, source_filename="low_confidence.mp3", confidence="0.89")],
    )

    summary = run_transcript_quality_backfill(
        TranscriptQualityBackfillConfig(
            database_url=settings.database_url,
            candidates_csv=candidates_csv,
            out_root=tmp_path / "dry_run",
            mode="dry-run",
        )
    )

    assert summary["planned_updates"] == 0
    assert summary["blocked_rows"] == 1
    blocked = (tmp_path / "dry_run" / "blocked_rows.csv").read_text(encoding="utf-8-sig")
    assert "llm_consensus_confidence_too_low" in blocked


def test_transcript_quality_backfill_uses_gpt_only_hard_gate_policy(tmp_path: Path) -> None:
    db_path = tmp_path / "calls.db"
    settings = replace(make_settings(), database_url=f"sqlite:///{db_path}")
    init_db(settings)
    session_factory = build_session_factory(settings)

    with session_factory() as session:
        allowed = CallRecord(
            source_file=str(tmp_path / "allowed.mp3"),
            source_filename="allowed.mp3",
            transcription_status="done",
            resolve_status="done",
            analysis_status="done",
            transcript_text="CLIENT: Абонент сейчас не может ответить. Оставьте сообщение.",
            analysis_json=_analysis("sales_call"),
        )
        blocked = CallRecord(
            source_file=str(tmp_path / "blocked.mp3"),
            source_filename="blocked.mp3",
            transcription_status="done",
            resolve_status="done",
            analysis_status="done",
            transcript_text="CLIENT: Абонент сейчас не может ответить. Оставьте сообщение.",
            analysis_json=_analysis("sales_call"),
        )
        session.add_all([allowed, blocked])
        session.commit()
        allowed_id = int(allowed.id)
        blocked_id = int(blocked.id)

    candidates_csv = tmp_path / "hard_gate_candidates.csv"
    _write_csv(
        candidates_csv,
        [
            _hard_gate_consensus_candidate(
                row_id=allowed_id,
                source_filename="allowed.mp3",
                consensus_queue="manual_review",
                policy_queue="gpt_auto_apply",
                gpt_decision="safe_apply",
                claude_decision="manual_review",
            ),
            _hard_gate_consensus_candidate(
                row_id=blocked_id,
                source_filename="blocked.mp3",
                consensus_queue="consensus_auto_apply",
                policy_queue="gpt_keep_current",
                gpt_decision="keep_current",
                claude_decision="safe_apply",
                review_decision="blocked_by_hard_gate_gpt_policy",
            ),
        ],
    )

    summary = run_transcript_quality_backfill(
        TranscriptQualityBackfillConfig(
            database_url=settings.database_url,
            candidates_csv=candidates_csv,
            out_root=tmp_path / "dry_run",
            mode="dry-run",
        )
    )

    assert summary["planned_updates"] == 1
    assert summary["blocked_rows"] == 1
    blocked_csv = (tmp_path / "dry_run" / "blocked_rows.csv").read_text(encoding="utf-8-sig")
    assert "hard_gate_policy_auto_apply_not_allowed" in blocked_csv
    assert "hard_gate_policy_queue_not_auto_apply" in blocked_csv
    assert "hard_gate_gpt_decision_not_safe_apply" in blocked_csv
