from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


def _safe_json(raw: str | None) -> dict:
    text = (raw or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize transcribe merge token usage from DB")
    parser.add_argument("--db", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    con = sqlite3.connect(str(Path(args.db)))
    cur = con.cursor()
    rows = cur.execute(
        """
        SELECT id, source_filename, transcription_status, transcript_variants_json
          FROM call_records
         ORDER BY id ASC
        """
    ).fetchall()

    total_calls = len(rows)
    transcription_done = 0
    transcription_failed = 0
    transcription_dead = 0
    merge_blocks = 0
    merge_blocks_with_tokens = 0
    merge_tokens_total = 0
    merge_durations_total = 0.0
    merge_calls_with_tokens: dict[int, int] = {}

    for call_id, source_filename, transcription_status, raw_payload in rows:
        status = str(transcription_status or "")
        if status == "done":
            transcription_done += 1
        elif status == "failed":
            transcription_failed += 1
        elif status == "dead":
            transcription_dead += 1

        payload = _safe_json(raw_payload)
        slots = ("manager", "client") if payload.get("mode") == "stereo" else ("full",)
        for slot in slots:
            block = payload.get(slot) or {}
            merge_meta = block.get("merge_meta") or {}
            if not isinstance(merge_meta, dict):
                continue
            provider = str(merge_meta.get("provider") or "")
            if provider != "codex_cli":
                continue
            merge_blocks += 1
            tokens = merge_meta.get("tokens_used_actual")
            duration_sec = merge_meta.get("duration_sec")
            if isinstance(tokens, int):
                merge_blocks_with_tokens += 1
                merge_tokens_total += tokens
                merge_calls_with_tokens.setdefault(int(call_id), 0)
                merge_calls_with_tokens[int(call_id)] += tokens
            if isinstance(duration_sec, (int, float)):
                merge_durations_total += float(duration_sec)

    summary = {
        "model": args.model,
        "db": str(Path(args.db)),
        "total_calls": total_calls,
        "transcription_done": transcription_done,
        "transcription_failed": transcription_failed,
        "transcription_dead": transcription_dead,
        "merge_blocks": merge_blocks,
        "merge_blocks_with_tokens": merge_blocks_with_tokens,
        "merge_tokens_total": merge_tokens_total,
        "avg_merge_tokens_per_block": round(merge_tokens_total / merge_blocks_with_tokens, 2)
        if merge_blocks_with_tokens
        else None,
        "avg_merge_tokens_per_call": round(
            sum(merge_calls_with_tokens.values()) / len(merge_calls_with_tokens), 2
        )
        if merge_calls_with_tokens
        else None,
        "merge_duration_total_sec": round(merge_durations_total, 2),
        "avg_merge_duration_per_block_sec": round(merge_durations_total / merge_blocks, 3)
        if merge_blocks
        else None,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
