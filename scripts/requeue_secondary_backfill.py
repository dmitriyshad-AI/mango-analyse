from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


def _is_missing_secondary(payload: dict) -> bool:
    mode = str(payload.get("mode") or "").strip()
    if mode == "stereo":
        manager = payload.get("manager") or {}
        client = payload.get("client") or {}
        return not str(manager.get("variant_b") or "").strip() or not str(
            client.get("variant_b") or ""
        ).strip()
    if mode == "mono_or_fallback":
        full = payload.get("full") or {}
        return not str(full.get("variant_b") or "").strip()
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--provider", default="gigaam")
    parser.add_argument("--only-exhausted", action="store_true")
    args = parser.parse_args()

    db_path = Path(args.db)
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    rows = cur.execute(
        """
        SELECT id, transcript_variants_json, last_error
          FROM call_records
         WHERE transcription_status = 'done'
           AND transcript_variants_json IS NOT NULL
        """
    ).fetchall()

    scanned = 0
    updated = 0
    exhausted_cleared = 0
    errors_cleared = 0
    claims_cleared = 0

    for call_id, raw_payload, last_error in rows:
        try:
            payload = json.loads(raw_payload or "{}")
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        if not _is_missing_secondary(payload):
            continue
        scanned += 1
        meta = payload.get("secondary_backfill_meta")
        if not isinstance(meta, dict):
            continue
        if str(meta.get("provider") or "").strip() != args.provider:
            continue
        if args.only_exhausted and not bool(meta.get("exhausted")):
            continue

        payload.pop("secondary_backfill_meta", None)
        new_last_error = last_error
        if isinstance(last_error, str) and last_error.startswith("backfill-second-asr:"):
            new_last_error = None
            errors_cleared += 1

        cur.execute(
            """
            UPDATE call_records
               SET transcript_variants_json = ?,
                   last_error = ?,
                   pipeline_stage = NULL,
                   pipeline_worker_id = NULL,
                   pipeline_claimed_at = NULL,
                   next_retry_at = NULL
             WHERE id = ?
            """,
            (json.dumps(payload, ensure_ascii=False), new_last_error, call_id),
        )
        updated += 1
        exhausted_cleared += 1
        claims_cleared += 1

    conn.commit()
    print(
        json.dumps(
            {
                "ok": True,
                "db": str(db_path),
                "provider": args.provider,
                "only_exhausted": bool(args.only_exhausted),
                "scanned_missing": scanned,
                "updated": updated,
                "exhausted_cleared": exhausted_cleared,
                "errors_cleared": errors_cleared,
                "claims_cleared": claims_cleared,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
