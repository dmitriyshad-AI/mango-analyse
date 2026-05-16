#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from mango_mvp.channels.telegram_pilot_store import guard_telegram_pilot_path


TELEGRAM_PILOT_EVAL_PACK_SCHEMA_VERSION = "telegram_pilot_eval_pack_v1"
DEFAULT_OUT_DIR = Path(".codex_local/telegram_pilot/eval_packs")


@dataclass(frozen=True)
class TelegramPilotEvalPackResult:
    out_dir: str
    private_full_text_output_path: str
    private_manual_review_path: str
    public_summary_path: str
    seed: int
    sample_size: int
    run_count: int
    dialogs_selected_total: int
    messages_selected_total: int

    def to_json_dict(self) -> Mapping[str, Any]:
        return {
            "schema_version": TELEGRAM_PILOT_EVAL_PACK_SCHEMA_VERSION,
            **asdict(self),
        }


def build_telegram_pilot_eval_pack(
    messages_jsonl: Path | str,
    out_dir: Path | str,
    *,
    seed: int,
    sample_size: int = 20,
    run_count: int = 1,
    min_messages: int = 4,
    private_full_text_output_path: Optional[Path | str] = None,
) -> TelegramPilotEvalPackResult:
    source = Path(messages_jsonl)
    target = Path(out_dir)
    guard_telegram_pilot_path(target)
    if private_full_text_output_path is not None:
        guard_telegram_pilot_path(private_full_text_output_path)
    rows = list(iter_jsonl_objects(source))
    dialogs = group_dialog_threads(rows)
    eligible = [thread for thread in dialogs.values() if dialog_is_eligible(thread, min_messages=min_messages)]
    eligible.sort(key=lambda thread: str(thread["dialog_id"]))

    rng = random.Random(seed)
    target.mkdir(parents=True, exist_ok=True)
    private_full_path = Path(private_full_text_output_path) if private_full_text_output_path else target / "private_dialog_threads.jsonl"
    private_review_path = target / "private_manual_review.csv"
    public_summary_path = target / "public_summary.json"

    selected_runs: list[Mapping[str, Any]] = []
    private_records: list[Mapping[str, Any]] = []
    for run_index in range(run_count):
        selected = sample_dialogs(eligible, sample_size=sample_size, rng=rng)
        run_id = f"run_{run_index + 1:02d}"
        private_records.extend(build_private_dialog_record(run_id, thread) for thread in selected)
        selected_runs.append(build_public_run_summary(run_id, selected, seed=seed))

    write_jsonl(private_full_path, private_records)
    write_manual_review_csv(private_review_path, private_records)
    public_summary = build_public_summary(
        source_path=source,
        private_full_path=private_full_path,
        private_review_path=private_review_path,
        seed=seed,
        sample_size=sample_size,
        run_count=run_count,
        min_messages=min_messages,
        eligible_count=len(eligible),
        runs=selected_runs,
    )
    assert_no_raw_text_in_public_summary(public_summary, private_records)
    public_summary_path.write_text(
        json.dumps(public_summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return TelegramPilotEvalPackResult(
        out_dir=str(target),
        private_full_text_output_path=str(private_full_path),
        private_manual_review_path=str(private_review_path),
        public_summary_path=str(public_summary_path),
        seed=seed,
        sample_size=sample_size,
        run_count=run_count,
        dialogs_selected_total=sum(int(run["dialogs_selected"]) for run in selected_runs),
        messages_selected_total=sum(int(run["messages_selected"]) for run in selected_runs),
    )


def iter_jsonl_objects(path: Path) -> Iterable[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            text = line.strip()
            if not text:
                continue
            parsed = json.loads(text)
            if not isinstance(parsed, dict):
                raise ValueError(f"JSONL row {line_number} must be an object")
            yield parsed


def group_dialog_threads(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Mapping[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        dialog_id = as_text(row.get("dialog_id") or row.get("chat_id") or row.get("channel_thread_id"))
        if not dialog_id:
            continue
        grouped.setdefault(dialog_id, []).append(dict(row))
    result: dict[str, Mapping[str, Any]] = {}
    for dialog_id, messages in grouped.items():
        sorted_messages = sorted(messages, key=message_sort_key)
        result[dialog_id] = {
            "dialog_id": dialog_id,
            "messages": sorted_messages,
            "topic_ids": sorted({topic for row in sorted_messages if (topic := public_topic_value(row))}),
            "manager_only": any(row_route(row) == "manager_only" for row in sorted_messages),
            "useful_feedback": any(row_useful_feedback(row) for row in sorted_messages),
        }
    return result


def dialog_is_eligible(thread: Mapping[str, Any], *, min_messages: int) -> bool:
    messages = list(thread.get("messages") or [])
    if len(messages) < min_messages:
        return False
    directions = {message_direction(row) for row in messages}
    return {"client", "manager"}.issubset(directions)


def sample_dialogs(
    eligible: Sequence[Mapping[str, Any]],
    *,
    sample_size: int,
    rng: random.Random,
) -> Sequence[Mapping[str, Any]]:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if len(eligible) <= sample_size:
        return tuple(eligible)
    indexes = rng.sample(range(len(eligible)), sample_size)
    return tuple(eligible[index] for index in sorted(indexes))


def build_private_dialog_record(run_id: str, thread: Mapping[str, Any]) -> Mapping[str, Any]:
    messages = list(thread.get("messages") or [])
    return {
        "schema_version": TELEGRAM_PILOT_EVAL_PACK_SCHEMA_VERSION,
        "run_id": run_id,
        "dialog_id": str(thread.get("dialog_id")),
        "message_count": len(messages),
        "client_message_count": sum(1 for row in messages if message_direction(row) == "client"),
        "manager_message_count": sum(1 for row in messages if message_direction(row) == "manager"),
        "topic_ids": list(thread.get("topic_ids") or []),
        "manager_only": bool(thread.get("manager_only")),
        "useful_feedback": bool(thread.get("useful_feedback")),
        "messages": [
            {
                "message_id": as_text(row.get("message_id") or row.get("channel_message_id")),
                "date": as_text(row.get("date") or row.get("received_at")),
                "direction": message_direction(row),
                "text": as_text(row.get("text")),
                "has_media": boolish(row.get("has_media")),
            }
            for row in messages
        ],
    }


def build_public_run_summary(run_id: str, selected: Sequence[Mapping[str, Any]], *, seed: int) -> Mapping[str, Any]:
    topic_counts: dict[str, int] = {}
    messages_selected = 0
    manager_only = 0
    useful_feedback = 0
    dialog_hashes: list[str] = []
    for thread in selected:
        dialog_hashes.append(public_dialog_hash(str(thread.get("dialog_id")), seed=seed))
        messages = list(thread.get("messages") or [])
        messages_selected += len(messages)
        if bool(thread.get("manager_only")):
            manager_only += 1
        if bool(thread.get("useful_feedback")):
            useful_feedback += 1
        topics = list(thread.get("topic_ids") or []) or ["unknown"]
        for topic in topics:
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
    dialogs_selected = len(selected)
    return {
        "run_id": run_id,
        "dialogs_selected": dialogs_selected,
        "messages_selected": messages_selected,
        "dialog_hashes": sorted(dialog_hashes),
        "topic_counts": dict(sorted(topic_counts.items())),
        "manager_only_count": manager_only,
        "manager_only_share": round(manager_only / dialogs_selected, 4) if dialogs_selected else 0.0,
        "useful_feedback_count": useful_feedback,
        "useful_feedback_share": round(useful_feedback / dialogs_selected, 4) if dialogs_selected else 0.0,
    }


def build_public_summary(
    *,
    source_path: Path,
    private_full_path: Path,
    private_review_path: Path,
    seed: int,
    sample_size: int,
    run_count: int,
    min_messages: int,
    eligible_count: int,
    runs: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    return {
        "schema_version": TELEGRAM_PILOT_EVAL_PACK_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_file_name": source_path.name,
        "seed": seed,
        "sample_size": sample_size,
        "run_count": run_count,
        "min_messages": min_messages,
        "eligible_dialogs": eligible_count,
        "dialogs_selected_total": sum(int(run["dialogs_selected"]) for run in runs),
        "messages_selected_total": sum(int(run["messages_selected"]) for run in runs),
        "private_full_text_output_path": str(private_full_path),
        "private_manual_review_path": str(private_review_path),
        "runs": list(runs),
        "safety": {
            "public_summary_contains_raw_text": False,
            "private_full_text_stays_local": True,
            "network_calls": False,
            "live_send": False,
            "write_crm": False,
            "write_tallanto": False,
            "write_stable_runtime": False,
        },
    }


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_manual_review_csv(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "run_id",
                "dialog_id",
                "message_count",
                "client_message_count",
                "manager_message_count",
                "topic_ids",
                "manager_only",
                "useful_feedback",
                "private_thread_text",
            ],
        )
        writer.writeheader()
        for record in records:
            thread_text = "\n".join(
                f"{message['date']} {message['direction']}: {message['text']}"
                for message in record.get("messages", [])
            )
            writer.writerow(
                {
                    "run_id": record["run_id"],
                    "dialog_id": record["dialog_id"],
                    "message_count": record["message_count"],
                    "client_message_count": record["client_message_count"],
                    "manager_message_count": record["manager_message_count"],
                    "topic_ids": ",".join(record.get("topic_ids") or []),
                    "manager_only": record["manager_only"],
                    "useful_feedback": record["useful_feedback"],
                    "private_thread_text": thread_text,
                }
            )


def assert_no_raw_text_in_public_summary(
    public_summary: Mapping[str, Any],
    private_records: Sequence[Mapping[str, Any]],
) -> None:
    public_blob = json.dumps(public_summary, ensure_ascii=False, sort_keys=True)
    raw_texts: set[str] = set()
    for record in private_records:
        for message in record.get("messages", []):
            text = as_text(message.get("text"))
            if len(text) >= 8:
                raw_texts.add(text)
    leaked = [text for text in raw_texts if text in public_blob]
    if leaked:
        raise ValueError("public summary contains raw Telegram text")


def message_sort_key(row: Mapping[str, Any]) -> tuple[str, int, str]:
    date_text = as_text(row.get("date") or row.get("received_at"))
    message_id = as_text(row.get("message_id") or row.get("channel_message_id"))
    try:
        message_num = int(message_id)
    except ValueError:
        message_num = 0
    return date_text, message_num, message_id


def message_direction(row: Mapping[str, Any]) -> str:
    if "out" in row:
        return "manager" if boolish(row.get("out")) else "client"
    direction = as_text(row.get("direction")).lower()
    if direction == "outbound":
        return "manager"
    return "client"


def row_route(row: Mapping[str, Any]) -> str:
    return as_text(row.get("route") or row.get("draft_route") or row.get("status")).lower()


def row_useful_feedback(row: Mapping[str, Any]) -> bool:
    text = as_text(row.get("feedback") or row.get("draft_status") or row.get("manager_feedback")).lower()
    return text in {"manager_marked_useful", "useful", "approved"}


def public_topic_value(row: Mapping[str, Any]) -> str:
    value = as_text(row.get("topic_id") or row.get("theme_id") or row.get("topic_key") or row.get("theme_key"))
    if not value:
        return ""
    allowed = set("abcdefghijklmnopqrstuvwxyz0123456789_.:-")
    lowered = value.lower()
    if any(ch not in allowed for ch in lowered):
        return "unknown"
    return lowered[:80]


def public_dialog_hash(dialog_id: str, *, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{dialog_id}".encode("utf-8")).hexdigest()[:16]


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = as_text(value).lower()
    return text in {"1", "true", "yes", "y", "out", "manager"}


def as_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build a private Telegram pilot eval pack and public safe summary.")
    parser.add_argument("--messages-jsonl", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--run-count", type=int, default=1)
    parser.add_argument("--min-messages", type=int, default=4)
    parser.add_argument("--private-full-text-output", type=Path)
    args = parser.parse_args(argv)

    result = build_telegram_pilot_eval_pack(
        args.messages_jsonl,
        args.out_dir,
        seed=args.seed,
        sample_size=args.sample_size,
        run_count=args.run_count,
        min_messages=args.min_messages,
        private_full_text_output_path=args.private_full_text_output,
    )
    print(json.dumps(result.to_json_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
