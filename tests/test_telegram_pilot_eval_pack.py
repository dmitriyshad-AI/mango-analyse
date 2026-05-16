from __future__ import annotations

import json
from pathlib import Path

from scripts.build_telegram_pilot_eval_pack import build_telegram_pilot_eval_pack


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def dialog_rows(dialog_id: int, *, topic_id: str = "theme:price", manager_only: bool = False) -> list[dict]:
    route = "manager_only" if manager_only else "draft_for_manager"
    return [
        {
            "dialog_id": dialog_id,
            "message_id": dialog_id * 10 + 1,
            "date": "2026-05-16T09:00:00+00:00",
            "text": f"client private text {dialog_id} a",
            "out": False,
            "topic_id": topic_id,
            "route": route,
        },
        {
            "dialog_id": dialog_id,
            "message_id": dialog_id * 10 + 2,
            "date": "2026-05-16T09:01:00+00:00",
            "text": f"manager private text {dialog_id} b",
            "out": True,
            "topic_id": topic_id,
            "route": route,
        },
        {
            "dialog_id": dialog_id,
            "message_id": dialog_id * 10 + 3,
            "date": "2026-05-16T09:02:00+00:00",
            "text": f"client private text {dialog_id} c",
            "out": False,
            "topic_id": topic_id,
            "route": route,
        },
        {
            "dialog_id": dialog_id,
            "message_id": dialog_id * 10 + 4,
            "date": "2026-05-16T09:03:00+00:00",
            "text": f"manager private text {dialog_id} d",
            "out": True,
            "topic_id": topic_id,
            "route": route,
            "draft_status": "manager_marked_useful",
        },
    ]


def test_eval_pack_samples_dialog_threads(tmp_path) -> None:
    messages_path = tmp_path / "messages.jsonl"
    rows = dialog_rows(1) + dialog_rows(2, topic_id="theme:schedule") + dialog_rows(3, manager_only=True)
    write_jsonl(messages_path, rows)

    result = build_telegram_pilot_eval_pack(messages_path, tmp_path / "pack", seed=11, sample_size=2)
    public_summary = json.loads(Path(result.public_summary_path).read_text(encoding="utf-8"))
    private_text = Path(result.private_full_text_output_path).read_text(encoding="utf-8")

    assert result.dialogs_selected_total == 2
    assert result.messages_selected_total == 8
    assert public_summary["runs"][0]["dialogs_selected"] == 2
    assert public_summary["runs"][0]["messages_selected"] == 8
    assert "client private text" in private_text
    assert Path(result.private_manual_review_path).exists()


def test_eval_pack_is_seed_reproducible(tmp_path) -> None:
    messages_path = tmp_path / "messages.jsonl"
    rows: list[dict] = []
    for dialog_id in range(1, 8):
        rows.extend(dialog_rows(dialog_id, topic_id=f"theme:{dialog_id:03d}"))
    write_jsonl(messages_path, rows)

    first = build_telegram_pilot_eval_pack(messages_path, tmp_path / "pack1", seed=42, sample_size=3, run_count=2)
    second = build_telegram_pilot_eval_pack(messages_path, tmp_path / "pack2", seed=42, sample_size=3, run_count=2)
    first_summary = json.loads(Path(first.public_summary_path).read_text(encoding="utf-8"))
    second_summary = json.loads(Path(second.public_summary_path).read_text(encoding="utf-8"))

    assert first_summary["runs"] == second_summary["runs"]


def test_public_summary_does_not_include_raw_text(tmp_path) -> None:
    messages_path = tmp_path / "messages.jsonl"
    secret_text = "client private text 777 a"
    rows = dialog_rows(777, topic_id="theme:docs") + dialog_rows(778, topic_id="theme:price")
    write_jsonl(messages_path, rows)

    result = build_telegram_pilot_eval_pack(messages_path, tmp_path / "pack", seed=5, sample_size=2)
    public_blob = Path(result.public_summary_path).read_text(encoding="utf-8")

    assert secret_text not in public_blob
    assert "manager private text" not in public_blob
    assert "private_full_text_output_path" in public_blob
