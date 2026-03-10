#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


def _row_text_chars(manager: str | None, client: str | None, full: str | None) -> int:
    manager_chars = len((manager or "").strip())
    client_chars = len((client or "").strip())
    if manager_chars or client_chars:
        return manager_chars + client_chars
    return len((full or "").strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Estimate LLM token budget for dual-ASR merge + structured analysis"
    )
    parser.add_argument("--database", required=True, help="Path to SQLite DB")
    parser.add_argument("--calls", type=int, default=100_000, help="Target number of calls")
    parser.add_argument(
        "--chars-per-token",
        type=float,
        default=2.7,
        help="Approximation factor for Russian text tokenization",
    )
    parser.add_argument(
        "--merge-rate",
        type=float,
        default=1.0,
        help="Share of calls sent to LLM merge (0..1). Example: 0.35 for selective merge.",
    )
    parser.add_argument(
        "--merge-overhead-tokens",
        type=float,
        default=300.0,
        help="Per-call prompt overhead for merge stage",
    )
    parser.add_argument(
        "--merge-output-overhead-tokens",
        type=float,
        default=40.0,
        help="Extra output tokens for merge JSON metadata",
    )
    parser.add_argument(
        "--analysis-overhead-tokens",
        type=float,
        default=250.0,
        help="Per-call prompt overhead for analysis stage",
    )
    parser.add_argument(
        "--analysis-output-tokens",
        type=float,
        default=220.0,
        help="Expected output tokens for structured analysis JSON",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=0,
        help="Ignore calls where derived transcript chars are below threshold",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    db_path = Path(args.database)
    if not db_path.exists():
        raise SystemExit(f"Database not found: {db_path}")

    merge_rate = max(0.0, min(1.0, args.merge_rate))
    cpt = max(1.0, args.chars_per_token)

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    rows = cur.execute(
        """
        SELECT transcript_manager, transcript_client, transcript_text
        FROM call_records
        WHERE transcription_status='done'
        """
    ).fetchall()

    text_chars_values: list[int] = []
    per_call_total_tokens: list[float] = []
    for manager, client, full in rows:
        chars = _row_text_chars(manager, client, full)
        if chars < args.min_chars:
            continue
        text_chars_values.append(chars)

        transcript_tokens = chars / cpt
        merge_tokens = (
            (2.0 * transcript_tokens + args.merge_overhead_tokens)
            + (transcript_tokens + args.merge_output_overhead_tokens)
        )
        analysis_tokens = (
            transcript_tokens + args.analysis_overhead_tokens + args.analysis_output_tokens
        )
        total_tokens = merge_rate * merge_tokens + analysis_tokens
        per_call_total_tokens.append(total_tokens)

    if not per_call_total_tokens:
        raise SystemExit("No transcribed rows for estimation (after filters)")

    avg_chars = sum(text_chars_values) / len(text_chars_values)
    avg_tokens_per_call = sum(per_call_total_tokens) / len(per_call_total_tokens)
    total_tokens = avg_tokens_per_call * args.calls

    payload = {
        "database": str(db_path),
        "rows_used": len(per_call_total_tokens),
        "calls_target": args.calls,
        "assumptions": {
            "chars_per_token": cpt,
            "merge_rate": merge_rate,
            "merge_overhead_tokens": args.merge_overhead_tokens,
            "merge_output_overhead_tokens": args.merge_output_overhead_tokens,
            "analysis_overhead_tokens": args.analysis_overhead_tokens,
            "analysis_output_tokens": args.analysis_output_tokens,
            "min_chars": args.min_chars,
        },
        "dataset_stats": {
            "avg_text_chars_per_call": round(avg_chars, 2),
        },
        "estimate": {
            "avg_tokens_per_call": round(avg_tokens_per_call, 2),
            "total_tokens": round(total_tokens, 2),
            "total_tokens_millions": round(total_tokens / 1_000_000.0, 2),
        },
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
