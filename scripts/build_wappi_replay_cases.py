#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from mango_mvp.replay_exam.exporter import RAW_ROOT, assert_raw_output_path
from mango_mvp.replay_exam.models import ReplayMessage
from mango_mvp.replay_exam.pseudonymizer import ReplayPseudonymizer, pii_signals
from mango_mvp.replay_exam.slicer import slice_teacher_forcing_cases


SCRUBBED_ROOT = Path("~/.mango_local/replay_exam/scrubbed").expanduser()


def _assert_scrubbed_output_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    root = SCRUBBED_ROOT.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"scrubbed replay output must stay under {root}") from exc
    return resolved


def _dialog_id(profile_id: str, chat_id: str) -> str:
    digest = hashlib.sha256(f"{profile_id}:{chat_id}".encode("utf-8")).hexdigest()[:16]
    return f"wappi_replay_{digest}"


def _message_from_scrubbed(raw: Mapping[str, Any]) -> ReplayMessage:
    return ReplayMessage(
        profile_id=str(raw.get("profile_id") or ""),
        chat_id=str(raw.get("chat_id") or ""),
        message_id=str(raw.get("message_id") or ""),
        text=str(raw.get("text") or ""),
        timestamp=int(raw.get("timestamp") or 0),
        from_me=bool(raw.get("from_me")),
        sender_name=str(raw.get("sender_name") or ""),
        raw=dict(raw.get("raw") or {}),
    )


def _message_to_json(message: ReplayMessage) -> Mapping[str, Any]:
    return {
        "profile_id": message.profile_id,
        "chat_id": message.chat_id,
        "message_id": message.message_id,
        "text": message.text,
        "timestamp": message.timestamp,
        "from_me": message.from_me,
        "sender_name": message.sender_name,
        "raw": dict(message.raw),
    }


def _case_to_json(case) -> Mapping[str, Any]:  # type: ignore[no-untyped-def]
    return {
        "dialog_id": case.dialog_id,
        "profile_id": case.profile_id,
        "chat_id": case.chat_id,
        "turn_id": case.turn_id,
        "brand": case.brand,
        "client_message": case.client_message,
        "manager_reference": case.manager_reference,
        "prefix_messages": [_message_to_json(item) for item in case.prefix_messages],
        "segment": case.segment,
        "expected_p0": case.expected_p0,
        "metadata": dict(case.metadata),
    }


def _iter_manifest_dialogs(manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for profile in manifest.get("profiles") or ():
        if not isinstance(profile, Mapping):
            continue
        for row in profile.get("selected_dialogs") or ():
            if isinstance(row, Mapping):
                rows.append(row)
    return rows


def build_cases_from_raw_manifest(
    *,
    raw_manifest_path: Path,
    out_root: Path = SCRUBBED_ROOT,
    max_dialogs: int = 10,
) -> Mapping[str, Any]:
    raw_manifest = json.loads(raw_manifest_path.expanduser().read_text(encoding="utf-8"))
    raw_root = RAW_ROOT.resolve()
    raw_manifest_resolved = raw_manifest_path.expanduser().resolve()
    try:
        raw_manifest_resolved.relative_to(raw_root)
    except ValueError as exc:
        raise ValueError(f"raw manifest must stay under {raw_root}") from exc

    out_dir = _assert_scrubbed_output_path(out_root / f"replay_cases_{raw_manifest_resolved.parent.name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    dialogs_dir = out_dir / "dialogs"
    dialogs_dir.mkdir(parents=True, exist_ok=True)
    cases_path = out_dir / "cases.jsonl"
    sample_path = out_dir / "human_review_sample_20_messages.json"

    cases: list[Mapping[str, Any]] = []
    sample_messages: list[Mapping[str, Any]] = []
    dialog_summaries: list[Mapping[str, Any]] = []
    leak_signals: list[Mapping[str, Any]] = []

    for row in _iter_manifest_dialogs(raw_manifest)[: max(1, max_dialogs)]:
        raw_file = Path(str(row.get("raw_file") or "")).expanduser()
        raw_file_resolved = raw_file.resolve()
        try:
            raw_file_resolved.relative_to(raw_root)
        except ValueError as exc:
            raise ValueError(f"raw dialog file must stay under {raw_root}: {raw_file}") from exc
        raw_payload = json.loads(raw_file_resolved.read_text(encoding="utf-8"))
        profile_id = str(raw_payload.get("profile_id") or row.get("profile_id") or "")
        chat_id = str(raw_payload.get("chat_id") or row.get("chat_id") or "")
        dialog_id = _dialog_id(profile_id, chat_id)
        pseudonymizer = ReplayPseudonymizer(dialog_salt=dialog_id)
        scrubbed_payload = pseudonymizer.object(raw_payload)
        signals = pii_signals(scrubbed_payload)
        if signals:
            leak_signals.append({"dialog_id": dialog_id, "signals": signals})
            continue
        messages = [_message_from_scrubbed(item) for item in scrubbed_payload.get("messages", ()) if isinstance(item, Mapping)]
        brand = str(row.get("brand") or "unknown")
        dialog_cases = slice_teacher_forcing_cases(messages, dialog_id=dialog_id, brand=brand)
        dialog_path = dialogs_dir / f"{dialog_id}.json"
        dialog_path.write_text(json.dumps(scrubbed_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        for message in messages:
            if len(sample_messages) < 20:
                sample_messages.append(
                    {
                        "dialog_id": dialog_id,
                        "from_me": message.from_me,
                        "text": message.text,
                        "timestamp": message.timestamp,
                    }
                )
        for case in dialog_cases:
            case_payload = _case_to_json(case)
            case_signals = pii_signals(case_payload)
            if case_signals:
                leak_signals.append({"dialog_id": dialog_id, "turn_id": case.turn_id, "signals": case_signals})
                continue
            cases.append(case_payload)
        dialog_summaries.append(
            {
                "dialog_id": dialog_id,
                "brand": brand,
                "message_count": len(messages),
                "case_count": len(dialog_cases),
                "segment_counts": {
                    segment: sum(1 for case in dialog_cases if case.segment == segment)
                    for segment in sorted({case.segment for case in dialog_cases})
                },
                "scrubbed_dialog": str(dialog_path),
            }
        )

    cases_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in cases),
        encoding="utf-8",
    )
    sample_path.write_text(json.dumps(sample_messages, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": "wappi_replay_scrubbed_manifest_v1",
        "raw_manifest": str(raw_manifest_resolved),
        "out_dir": str(out_dir),
        "cases_jsonl": str(cases_path),
        "human_review_sample": str(sample_path),
        "dialog_count": len(dialog_summaries),
        "case_count": len(cases),
        "leak_count": len(leak_signals),
        "leaks": leak_signals,
        "dialogs": dialog_summaries,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if leak_signals:
        raise RuntimeError(f"scrubbed replay contains PII signals: {leak_signals[:3]}")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Pseudonymize raw Wappi replay dumps and build teacher-forcing cases.")
    parser.add_argument("--raw-manifest", required=True, type=Path)
    parser.add_argument("--out-root", type=Path, default=SCRUBBED_ROOT)
    parser.add_argument("--max-dialogs", type=int, default=10)
    args = parser.parse_args()

    manifest = build_cases_from_raw_manifest(
        raw_manifest_path=args.raw_manifest,
        out_root=args.out_root,
        max_dialogs=args.max_dialogs,
    )
    print(f"scrubbed_manifest={manifest['out_dir']}/manifest.json")
    print(f"case_count={manifest['case_count']}")
    print(f"leak_count={manifest['leak_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
