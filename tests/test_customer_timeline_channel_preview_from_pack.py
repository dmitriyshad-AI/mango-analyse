from __future__ import annotations

import json
import os
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mango_mvp.customer_timeline import (
    CustomerTimelineApprovedContextPackConfig,
    CustomerTimelineChannelPreviewFromPackConfig,
    build_customer_timeline_approved_context_pack,
    build_customer_timeline_channel_preview_from_pack,
)
from mango_mvp.customer_timeline.channel_preview_from_pack import main
from tests.test_customer_timeline_approved_context_pack import build_approval_artifacts


FIXED_TIME = datetime(2026, 5, 13, 12, 0, tzinfo=timezone.utc)


def test_channel_preview_from_pack_builds_manager_draft_without_live_send(tmp_path: Path) -> None:
    _db_path, pack_path = build_context_pack_file(tmp_path)
    message_path = write_inbound_message(
        tmp_path,
        text="Сколько стоит подготовка к ЕГЭ? Мой телефон +79161234567, email parent@example.com",
    )
    out_preview = tmp_path / "preview" / "manager_draft.json"

    report = build_customer_timeline_channel_preview_from_pack(
        config=CustomerTimelineChannelPreviewFromPackConfig(
            allowed_root=tmp_path,
            context_pack_json=pack_path,
            inbound_message_json=message_path,
            out_preview_json=out_preview,
        ),
        generated_at=FIXED_TIME,
    )
    serialized = json.dumps(report, ensure_ascii=False, sort_keys=True)
    draft = report["draft_preview"]
    actions = {action["action_type"]: action for action in draft["reply"]["recommended_actions"]}

    assert json.loads(out_preview.read_text(encoding="utf-8")) == report
    assert report["validation_ok"] is True
    assert report["status"] == "draft_ready_for_manager_review"
    assert report["summary"]["draft_created"] is True
    assert report["summary"]["can_send"] is False
    assert report["summary"]["requires_manager_review"] is True
    assert draft["draft_id"].startswith("channel_pack_draft:")
    assert draft["idempotency_key"].startswith("channel_pack_preview:")
    assert draft["reply"]["requires_approval"] is True
    assert "стоимости" in draft["reply"]["text"]
    assert draft["reply"]["metadata"]["customer_context_used"] is True
    assert actions["draft_client_message"]["payload"]["approved_context_pack_id"] == report["source_refs"]["context_pack_id"]
    assert actions["draft_client_message"]["payload"]["live_send_enabled"] is False
    assert actions["mark_manual_review"]["payload"]["live_send_enabled"] is False
    assert report["safety"]["live_send"] is False
    assert report["safety"]["write_crm"] is False
    assert report["safety"]["write_tallanto"] is False
    assert report["safety"]["network_calls"] is False
    assert report["safety"]["subprocess_calls"] is False
    assert report["safety"]["llm_calls"] is False
    assert report["safety"]["rag_used"] is False
    assert "raw_payload" not in serialized
    assert "provider_raw_payload" not in serialized
    assert "record_json" not in serialized
    assert "/not/read/transcript.json" not in serialized
    assert "/secret/audio.mp3" not in serialized
    assert "+79161234567" not in serialized
    assert "parent@example.com" not in serialized
    assert str(pack_path) not in serialized
    assert str(message_path) not in serialized


def test_channel_preview_from_pack_blocks_rejected_or_invalid_context_pack(tmp_path: Path) -> None:
    _db_path, pack_path = build_context_pack_file(tmp_path, decision="reject")
    message_path = write_inbound_message(tmp_path)

    report = build_customer_timeline_channel_preview_from_pack(
        config=CustomerTimelineChannelPreviewFromPackConfig(
            allowed_root=tmp_path,
            context_pack_json=pack_path,
            inbound_message_json=message_path,
        ),
        generated_at=FIXED_TIME,
    )

    assert report["validation_ok"] is False
    assert report["status"] == "blocked"
    assert report["draft_preview"] is None
    assert "context_pack_not_valid" in report["summary"]["blocked_reasons"]
    assert "context_pack_not_approved:blocked" in report["summary"]["blocked_reasons"]


def test_channel_preview_from_pack_blocks_non_inbound_message(tmp_path: Path) -> None:
    _db_path, pack_path = build_context_pack_file(tmp_path)
    message_path = write_inbound_message(tmp_path, direction="outbound")

    report = build_customer_timeline_channel_preview_from_pack(
        config=CustomerTimelineChannelPreviewFromPackConfig(
            allowed_root=tmp_path,
            context_pack_json=pack_path,
            inbound_message_json=message_path,
        ),
        generated_at=FIXED_TIME,
    )

    assert report["validation_ok"] is False
    assert report["draft_preview"] is None
    assert "message_direction_not_inbound" in report["summary"]["blocked_reasons"]


def test_channel_preview_from_pack_blocks_poisoned_context_pack(tmp_path: Path) -> None:
    _db_path, pack_path = build_context_pack_file(tmp_path)
    pack = json.loads(pack_path.read_text(encoding="utf-8"))
    pack["raw_payload"] = {"token": "secret"}
    pack["approved_context"]["items"][0]["customer_id"] = "customer:internal"
    pack_path.write_text(json.dumps(pack, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    message_path = write_inbound_message(tmp_path)

    report = build_customer_timeline_channel_preview_from_pack(
        config=CustomerTimelineChannelPreviewFromPackConfig(
            allowed_root=tmp_path,
            context_pack_json=pack_path,
            inbound_message_json=message_path,
        ),
        generated_at=FIXED_TIME,
    )

    assert report["validation_ok"] is False
    assert report["draft_preview"] is None
    assert "context_pack_approved_items_not_safe" in report["summary"]["blocked_reasons"]
    assert "context_pack_contains_forbidden_markers" in report["summary"]["blocked_reasons"]
    assert "raw_payload" not in json.dumps(report, ensure_ascii=False, sort_keys=True)


def test_channel_preview_from_pack_cli_is_artifact_only_no_network_or_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("channel preview from pack must not use network/subprocess")

    monkeypatch.setattr(subprocess, "run", fail)
    monkeypatch.setattr(subprocess, "Popen", fail)
    monkeypatch.setattr(os, "system", fail)
    monkeypatch.setattr(socket, "socket", fail)
    _db_path, pack_path = build_context_pack_file(tmp_path)
    message_path = write_inbound_message(tmp_path, text="Пожалуйста, перезвоните мне завтра")
    out_preview = tmp_path / "preview" / "cli_preview.json"

    rc = main(
        [
            "--allowed-root",
            str(tmp_path),
            "--context-pack-json",
            str(pack_path),
            "--inbound-message-json",
            str(message_path),
            "--out-preview-json",
            str(out_preview),
        ]
    )
    report = json.loads(out_preview.read_text(encoding="utf-8"))

    assert rc == 0
    assert report["validation_ok"] is True
    assert report["safety"]["network_calls"] is False
    assert report["safety"]["subprocess_calls"] is False
    assert report["safety"]["write_runtime_db"] is False


def test_channel_preview_from_pack_cli_returns_one_for_business_blocker(tmp_path: Path) -> None:
    _db_path, pack_path = build_context_pack_file(tmp_path)
    message_path = write_inbound_message(tmp_path, direction="internal")
    out_preview = tmp_path / "preview" / "blocked_preview.json"

    rc = main(
        [
            "--allowed-root",
            str(tmp_path),
            "--context-pack-json",
            str(pack_path),
            "--inbound-message-json",
            str(message_path),
            "--out-preview-json",
            str(out_preview),
        ]
    )
    report = json.loads(out_preview.read_text(encoding="utf-8"))

    assert rc == 1
    assert report["validation_ok"] is False
    assert "message_direction_not_inbound" in report["summary"]["blocked_reasons"]


def test_channel_preview_from_pack_path_guards(tmp_path: Path) -> None:
    _db_path, pack_path = build_context_pack_file(tmp_path)
    message_path = write_inbound_message(tmp_path)
    stable_input = tmp_path / "stable_runtime" / "message.json"
    stable_input.parent.mkdir()
    stable_input.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="stable_runtime"):
        CustomerTimelineChannelPreviewFromPackConfig(
            allowed_root=tmp_path,
            context_pack_json=pack_path,
            inbound_message_json=stable_input,
        )
    with pytest.raises(ValueError, match="allowed root"):
        CustomerTimelineChannelPreviewFromPackConfig(
            allowed_root=tmp_path,
            context_pack_json=pack_path,
            inbound_message_json=tmp_path.parent / "outside_message.json",
        )
    with pytest.raises(ValueError, match="overwrite context pack JSON"):
        CustomerTimelineChannelPreviewFromPackConfig(
            allowed_root=tmp_path,
            context_pack_json=pack_path,
            inbound_message_json=message_path,
            out_preview_json=pack_path,
        )
    with pytest.raises(ValueError, match="overwrite inbound message JSON"):
        CustomerTimelineChannelPreviewFromPackConfig(
            allowed_root=tmp_path,
            context_pack_json=pack_path,
            inbound_message_json=message_path,
            out_preview_json=message_path,
        )
    with pytest.raises(ValueError, match="stable_runtime"):
        CustomerTimelineChannelPreviewFromPackConfig(
            allowed_root=tmp_path,
            context_pack_json=pack_path,
            inbound_message_json=message_path,
            out_preview_json=tmp_path / "stable_runtime" / "preview.json",
        )


def test_channel_preview_from_pack_does_not_mutate_stable_runtime(tmp_path: Path) -> None:
    _db_path, pack_path = build_context_pack_file(tmp_path)
    message_path = write_inbound_message(tmp_path)
    stable_dir = tmp_path / "stable_runtime"
    stable_dir.mkdir()
    sentinel = stable_dir / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")
    before_stable = sorted(path.relative_to(stable_dir) for path in stable_dir.rglob("*"))

    report = build_customer_timeline_channel_preview_from_pack(
        config=CustomerTimelineChannelPreviewFromPackConfig(
            allowed_root=tmp_path,
            context_pack_json=pack_path,
            inbound_message_json=message_path,
            out_preview_json=tmp_path / "preview" / "safe.json",
        ),
        generated_at=FIXED_TIME,
    )

    assert report["validation_ok"] is True
    assert sorted(path.relative_to(stable_dir) for path in stable_dir.rglob("*")) == before_stable
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_channel_preview_from_pack_is_deterministic_and_pack_specific(tmp_path: Path) -> None:
    _db_path, pack_path = build_context_pack_file(tmp_path)
    message_path = write_inbound_message(tmp_path)
    config = CustomerTimelineChannelPreviewFromPackConfig(
        allowed_root=tmp_path,
        context_pack_json=pack_path,
        inbound_message_json=message_path,
    )
    first = build_customer_timeline_channel_preview_from_pack(config=config, generated_at=FIXED_TIME)
    second = build_customer_timeline_channel_preview_from_pack(config=config, generated_at=FIXED_TIME)
    pack = json.loads(pack_path.read_text(encoding="utf-8"))
    pack["pack_id"] = "approved_context_pack:changed"
    changed_pack_path = tmp_path / "approved" / "changed_pack.json"
    changed_pack_path.write_text(json.dumps(pack, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    changed = build_customer_timeline_channel_preview_from_pack(
        config=CustomerTimelineChannelPreviewFromPackConfig(
            allowed_root=tmp_path,
            context_pack_json=changed_pack_path,
            inbound_message_json=message_path,
        ),
        generated_at=FIXED_TIME,
    )

    assert first == second
    assert first["preview_id"] != changed["preview_id"]
    assert first["draft_preview"]["draft_id"] != changed["draft_preview"]["draft_id"]
    assert first["draft_preview"]["idempotency_key"] != changed["draft_preview"]["idempotency_key"]


def build_context_pack_file(tmp_path: Path, *, decision: str = "approve") -> tuple[Path, Path]:
    db_path, _customer_id, workspace_path, decisions_path, report_path = build_approval_artifacts(tmp_path, decision=decision)
    out_pack = tmp_path / "approved" / f"{decision}_context_pack.json"
    build_customer_timeline_approved_context_pack(
        config=CustomerTimelineApprovedContextPackConfig(
            timeline_db=db_path,
            allowed_root=tmp_path,
            workspace_json=workspace_path,
            decisions_jsonl=decisions_path,
            approval_report_json=report_path,
            out_pack_json=out_pack,
        ),
        generated_at=FIXED_TIME,
    )
    assert out_pack.exists()
    return db_path, out_pack


def write_inbound_message(
    tmp_path: Path,
    *,
    text: str = "Здравствуйте, можно записаться на консультацию?",
    direction: str = "inbound",
) -> Path:
    path = tmp_path / "messages" / f"{direction}_message.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "channel": "site_chat",
        "channel_message_id": "msg-1",
        "channel_thread_id": "thread-1",
        "channel_user_id": "visitor-1",
        "direction": direction,
        "text": text,
        "received_at": "2026-05-13T12:05:00+03:00",
        "raw_payload": {"token": "secret"},
        "metadata": {
            "source_path": "/secret/message.json",
            "audio_path": "/not/read/audio.mp3",
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
