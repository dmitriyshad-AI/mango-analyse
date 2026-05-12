from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

from mango_mvp.db import build_session_factory, init_db
from mango_mvp.models import CallRecord
from mango_mvp.quality.transcript_quality_guardrails_dry_run import (
    GuardrailsDryRunConfig,
    build_transcript_quality_guardrails_dry_run,
)
from tests.test_dialogue_format import make_settings


def _analysis(call_type: str, *, history: str = "", next_step: str | None = None) -> str:
    return json.dumps(
        {
            "history_summary": history,
            "structured_fields": {
                "interests": {"products": [], "subjects": []},
                "objections": [],
                "next_step": {"action": next_step},
            },
            "quality_flags": {"call_type": call_type},
            "next_step": next_step,
        },
        ensure_ascii=False,
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def test_build_transcript_quality_guardrails_dry_run_reports_core_queues(tmp_path: Path) -> None:
    db_path = tmp_path / "calls.db"
    settings = replace(make_settings(), database_url=f"sqlite:///{db_path}")
    init_db(settings)
    session_factory = build_session_factory(settings)

    with session_factory() as session:
        session.add_all(
            [
                CallRecord(
                    source_file=str(tmp_path / "voicemail.mp3"),
                    source_filename="voicemail.mp3",
                    transcription_status="done",
                    resolve_status="done",
                    analysis_status="done",
                    duration_sec=25,
                    transcript_text=(
                        "MANAGER:\nДобрый день.\n\n"
                        "CLIENT:\nЗвонок был перенаправлен на голосовой почтовый ящик. "
                        "Оставьте сообщение после звукового сигнала. Продолжение следует."
                    ),
                    analysis_json=_analysis("technical_call", history="Абонент недоступен."),
                ),
                CallRecord(
                    source_file=str(tmp_path / "live_service.mp3"),
                    source_filename="live_service.mp3",
                    transcription_status="done",
                    resolve_status="done",
                    analysis_status="done",
                    duration_sec=180,
                    transcript_text=(
                        "MANAGER:\nДобрый день, я отправлю чек на почту и завтра перезвоню.\n\n"
                        "CLIENT:\nДа, чек нужен на почту. Оплату внесли, но ссылка на занятие не работает, "
                        "помогите с доступом и расписанием."
                    ),
                    analysis_json=_analysis(
                        "service_call",
                        history="Клиент подтвердил оплату и попросил помочь с доступом к занятию.",
                        next_step="Перезвонить клиенту",
                    ),
                ),
                CallRecord(
                    source_file=str(tmp_path / "borderline.mp3"),
                    source_filename="borderline.mp3",
                    transcription_status="done",
                    resolve_status="done",
                    analysis_status="done",
                    duration_sec=55,
                    transcript_text=(
                        "MANAGER:\nДобрый день, учебный центр.\n\n"
                        "CLIENT:\nАбонент сейчас не может ответить на ваш звонок. "
                        "Попробуйте перезвонить позднее."
                    ),
                    analysis_json=_analysis("service_call", next_step="Перезвонить клиенту"),
                ),
            ]
        )
        session.commit()

    out_root = tmp_path / "dry_run"
    summary = build_transcript_quality_guardrails_dry_run(
        GuardrailsDryRunConfig(
            database_url=settings.database_url,
            out_root=out_root,
        )
    )

    assert summary["scanned_rows"] == 3
    assert summary["eligible_transcripts"] == 3
    assert summary["auto_fix_candidates"] == 2
    assert summary["manual_review_candidates"] == 0
    assert summary["protected_live_dialogues"] == 1
    assert summary["contentful_auto_fix_candidates"] == 2
    assert summary["label_counts"]["non_conversation_high_confidence"] == 2
    assert summary["label_counts"]["contentful_protected_live_dialogue"] == 1

    auto_fix_rows = _read_csv(out_root / "auto_fix_candidates.csv")
    manual_rows = _read_csv(out_root / "manual_review_candidates.csv")
    protected_rows = _read_csv(out_root / "protected_live_sample.csv")
    all_rows = _read_csv(out_root / "guardrails_all_results.csv")

    assert [row["source_filename"] for row in auto_fix_rows] == ["voicemail.mp3", "borderline.mp3"]
    assert manual_rows == []
    assert [row["source_filename"] for row in protected_rows] == ["live_service.mp3"]
    assert len(all_rows) == 3
    assert (out_root / "summary.json").exists()
    assert (out_root / "TRANSCRIPT_QUALITY_GUARDRAILS_DRY_RUN.md").exists()
    assert (out_root / "monthly_summary.csv").exists()
    assert (out_root / "call_type_summary.csv").exists()
