#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from mango_mvp.config import get_settings
from mango_mvp.services.transcribe import TranscribeService


def _load_candidates(
    db_path: Path,
    limit: int,
    *,
    max_chars: int,
    min_chars: int,
    similarity_threshold: float,
) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT id, source_filename, transcript_variants_json
            FROM call_records
            WHERE transcription_status='done' AND transcript_variants_json IS NOT NULL
            ORDER BY id
            """
        ).fetchall()
    finally:
        conn.close()

    settings = get_settings()
    helper = TranscribeService(settings)
    out: List[Dict[str, Any]] = []
    for row in rows:
        try:
            payload = json.loads(str(row["transcript_variants_json"] or ""))
        except Exception:
            continue
        if str(payload.get("mode") or "") != "stereo":
            continue
        manager = payload.get("manager")
        if not isinstance(manager, dict):
            continue
        a = str(manager.get("variant_a") or "").strip()
        b = str(manager.get("variant_b") or "").strip()
        if not a or not b:
            continue
        a = a[:max_chars].strip()
        b = b[:max_chars].strip()
        if len(a) < min_chars or len(b) < min_chars:
            continue
        similarity = helper._similarity_ratio(a, b)
        # Force real merge call path (avoid skip_high_similarity fast path).
        if similarity >= similarity_threshold:
            continue
        out.append(
            {
                "call_id": int(row["id"]),
                "source_filename": str(row["source_filename"] or ""),
                "variant_a": a,
                "variant_b": b,
                "similarity": round(similarity, 4),
            }
        )
        if len(out) >= limit:
            break
    return out


def _run_model(
    model_name: str,
    candidates: List[Dict[str, Any]],
    *,
    timeout_sec: int,
) -> Dict[str, Any]:
    base = get_settings()
    settings = replace(
        base,
        dual_merge_provider="codex_cli",
        codex_merge_model=model_name,
        codex_cli_timeout_sec=max(5, int(timeout_sec)),
    )
    svc = TranscribeService(settings)

    items: List[Dict[str, Any]] = []
    started = time.time()
    total = len(candidates)
    print(f"[{model_name}] start jobs={total}", flush=True)
    for idx, call in enumerate(candidates, start=1):
        t0 = time.time()
        result = svc._merge_variant_pair(
            str(call["variant_a"]),
            str(call["variant_b"]),
            speaker_label="Менеджер",
        )
        elapsed = round(time.time() - t0, 3)
        provider = str(result.get("provider") or "")
        notes = str(result.get("notes") or "")
        items.append(
            {
                "call_id": int(call["call_id"]),
                "source_filename": str(call["source_filename"]),
                "provider": provider,
                "selection": str(result.get("selection") or ""),
                "confidence": result.get("confidence"),
                "similarity": call["similarity"],
                "elapsed_sec": elapsed,
                "notes": notes[:400],
            }
        )
        print(
            f"[{model_name}] {idx}/{total} provider={provider} elapsed={elapsed}s",
            flush=True,
        )

    providers: Dict[str, int] = {}
    elapsed_values = [float(it["elapsed_sec"]) for it in items]
    for it in items:
        providers[it["provider"]] = providers.get(it["provider"], 0) + 1
    codex_conf = [
        float(it["confidence"])
        for it in items
        if it["provider"] == "codex_cli" and isinstance(it["confidence"], (int, float))
    ]

    summary = {
        "model": model_name,
        "jobs": len(items),
        "providers": providers,
        "codex_ok": providers.get("codex_cli", 0),
        "fallback_rule": providers.get("rule_fallback", 0),
        "other": sum(v for k, v in providers.items() if k not in {"codex_cli", "rule_fallback"}),
        "total_elapsed_sec": round(time.time() - started, 3),
        "avg_elapsed_sec": round(sum(elapsed_values) / len(elapsed_values), 3)
        if elapsed_values
        else 0.0,
        "avg_codex_confidence": round(sum(codex_conf) / len(codex_conf), 4)
        if codex_conf
        else None,
    }
    return {"summary": summary, "items": items}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare Codex CLI merge models on the same dual-ASR call sample."
    )
    parser.add_argument("--db", required=True, help="Path to sqlite db with call_records")
    parser.add_argument("--calls", type=int, default=50, help="Number of calls to benchmark")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gpt-5.3-codex", "gpt-5.4"],
        help="Model names for CODEX_MERGE_MODEL benchmark",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=22,
        help="Per-call codex_cli timeout seconds",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=120,
        help="Max chars per variant (speed/quality tradeoff for benchmark)",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=40,
        help="Min chars per variant to include",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.985,
        help="Skip pairs above this similarity to force real merge calls",
    )
    parser.add_argument("--out", required=True, help="Output json report path")
    args = parser.parse_args()

    db_path = Path(args.db).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    candidates = _load_candidates(
        db_path,
        max(1, int(args.calls)),
        max_chars=max(1, int(args.max_chars)),
        min_chars=max(1, int(args.min_chars)),
        similarity_threshold=float(args.similarity_threshold),
    )
    if not candidates:
        raise SystemExit("No suitable dual-variant candidates found for benchmark.")

    started = time.time()
    by_model: Dict[str, Any] = {}
    for model in args.models:
        by_model[str(model)] = _run_model(
            str(model),
            candidates,
            timeout_sec=max(1, int(args.timeout_sec)),
        )

    report = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "db": str(db_path),
        "calls_requested": int(args.calls),
        "calls_benchmarked": len(candidates),
        "models": args.models,
        "timeout_sec": int(args.timeout_sec),
        "max_chars": int(args.max_chars),
        "candidates": [
            {
                "call_id": c["call_id"],
                "source_filename": c["source_filename"],
                "similarity": c["similarity"],
            }
            for c in candidates
        ],
        "results": by_model,
        "total_elapsed_sec": round(time.time() - started, 3),
    }
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    compact = {
        "calls_benchmarked": report["calls_benchmarked"],
        "models": {
            model: payload["summary"] for model, payload in by_model.items()
        },
        "total_elapsed_sec": report["total_elapsed_sec"],
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
