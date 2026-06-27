from __future__ import annotations

import argparse
import json
import re
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import tiktoken

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mango_mvp.utils.codex_cli import append_codex_service_tier


PROMPT_TEMPLATE = """You merge two ASR transcript variants for the same speaker in one phone call.
Rules:
1) Use only information from variants A and B. Do not invent facts.
2) Keep chronology and intent. Remove obvious ASR loops and garbage.
3) If uncertain, prefer A.
4) Return strict JSON with keys:
   - merged_text (string)
   - selection (one of: A, B, MIX)
   - confidence (number 0..1)
   - notes (string)
Return a single-line minified JSON object. No markdown, no extra keys.

Speaker: {speaker_label}

Variant A:
{variant_a}

Variant B:
{variant_b}
"""

TOKENS_USED_RE = re.compile(r"tokens used\s*([0-9\s\u00a0]+)", re.I)
ENCODING = tiktoken.get_encoding("o200k_base")


def _tok_count(text: str) -> int:
    return len(ENCODING.encode(text))


def _parse_tokens_used(stderr: str) -> int | None:
    match = TOKENS_USED_RE.search(stderr or "")
    if not match:
        return None
    digits = "".join(ch for ch in match.group(1) if ch.isdigit())
    return int(digits) if digits else None


def _extract_json(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    if raw.startswith("{") and raw.endswith("}"):
        return json.loads(raw)
    start = raw.find("{")
    while start != -1:
        depth = 0
        for idx in range(start, len(raw)):
            ch = raw[idx]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(raw[start : idx + 1])
        start = raw.find("{", start + 1)
    raise RuntimeError("response does not contain JSON object")


def _safe_json(raw: str | None) -> dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _speaker_label(slot: str) -> str:
    return {
        "manager": "Менеджер",
        "client": "Клиент",
        "full": "Полный звонок",
    }.get(slot, slot)


def _load_selection(db_path: Path, selection_path: Path, sample_size: int) -> list[dict[str, Any]]:
    if selection_path.exists():
        payload = json.loads(selection_path.read_text(encoding="utf-8"))
        return list(payload.get("records") or [])

    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    rows = cur.execute(
        """
        SELECT id, source_filename, started_at, transcript_variants_json
          FROM call_records
         WHERE transcript_variants_json IS NOT NULL
           AND transcript_variants_json != ''
         ORDER BY started_at DESC, id DESC
        """
    ).fetchall()

    selected: list[dict[str, Any]] = []
    for row in rows:
        payload = _safe_json(row["transcript_variants_json"])
        if not payload.get("secondary_provider"):
            continue
        slots = ("manager", "client") if payload.get("mode") == "stereo" else ("full",)
        blocks: list[dict[str, Any]] = []
        for slot in slots:
            block = payload.get(slot) or {}
            variant_a = str(block.get("variant_a") or "").strip()
            variant_b = str(block.get("variant_b") or "").strip()
            if not variant_a or not variant_b:
                continue
            prompt = PROMPT_TEMPLATE.format(
                speaker_label=_speaker_label(slot),
                variant_a=variant_a,
                variant_b=variant_b,
            )
            blocks.append(
                {
                    "slot": slot,
                    "speaker_label": _speaker_label(slot),
                    "prompt_tokens_est": _tok_count(prompt),
                }
            )
        if not blocks:
            continue
        selected.append(
            {
                "call_id": int(row["id"]),
                "source_filename": str(row["source_filename"] or ""),
                "started_at": row["started_at"],
                "block_count": len(blocks),
                "blocks": blocks,
            }
        )
        if len(selected) >= sample_size:
            break

    summary = {
        "db_path": str(db_path),
        "sample_size_records": len(selected),
        "sample_size_blocks": sum(int(item["block_count"]) for item in selected),
        "records": selected,
    }
    selection_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return selected


def _fetch_record_blocks(db_path: Path, call_id: int) -> tuple[str, list[dict[str, Any]]]:
    con = sqlite3.connect(str(db_path))
    cur = con.cursor()
    row = cur.execute(
        "SELECT source_filename, transcript_variants_json FROM call_records WHERE id = ?",
        (call_id,),
    ).fetchone()
    if row is None:
        raise RuntimeError(f"call_id={call_id} not found")
    source_filename = str(row[0] or "")
    payload = _safe_json(row[1])
    slots = ("manager", "client") if payload.get("mode") == "stereo" else ("full",)
    blocks: list[dict[str, Any]] = []
    for slot in slots:
        block = payload.get(slot) or {}
        variant_a = str(block.get("variant_a") or "").strip()
        variant_b = str(block.get("variant_b") or "").strip()
        if not variant_a or not variant_b:
            continue
        prompt = PROMPT_TEMPLATE.format(
            speaker_label=_speaker_label(slot),
            variant_a=variant_a,
            variant_b=variant_b,
        )
        blocks.append(
            {
                "slot": slot,
                "speaker_label": _speaker_label(slot),
                "prompt": prompt,
                "prompt_tokens_est": _tok_count(prompt),
            }
        )
    return source_filename, blocks


def _load_completed(results_path: Path) -> set[tuple[int, str]]:
    completed: set[tuple[int, str]] = set()
    if not results_path.exists():
        return completed
    for line in results_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        call_id = payload.get("call_id")
        slot = payload.get("slot")
        if isinstance(call_id, int) and isinstance(slot, str):
            completed.add((call_id, slot))
    return completed


def _write_progress(progress_path: Path, payload: dict[str, Any]) -> None:
    progress_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _build_summary(model: str, rows: list[dict[str, Any]], *, sample_records: int) -> dict[str, Any]:
    ok_rows = [row for row in rows if row.get("returncode") == 0]
    token_rows = [int(row["tokens_used_actual"]) for row in ok_rows if row.get("tokens_used_actual") is not None]
    prompt_rows = [int(row["prompt_tokens_est"]) for row in ok_rows if row.get("prompt_tokens_est") is not None]
    output_rows = [int(row["output_tokens_est"]) for row in ok_rows if row.get("output_tokens_est") is not None]
    duration_rows = [float(row["duration_sec"]) for row in ok_rows if row.get("duration_sec") is not None]

    per_record_tokens: dict[int, int] = {}
    for row in ok_rows:
        call_id = int(row["call_id"])
        per_record_tokens.setdefault(call_id, 0)
        per_record_tokens[call_id] += int(row.get("tokens_used_actual") or 0)

    return {
        "model": model,
        "sample_size_records": sample_records,
        "sample_size_blocks": len(rows),
        "successful_blocks": len(ok_rows),
        "failed_blocks": len(rows) - len(ok_rows),
        "tokens_used_actual_total": sum(token_rows),
        "tokens_used_actual_avg_per_block": round(statistics.mean(token_rows), 2) if token_rows else None,
        "tokens_used_actual_avg_per_record": round(statistics.mean(per_record_tokens.values()), 2)
        if per_record_tokens
        else None,
        "prompt_tokens_est_total": sum(prompt_rows),
        "prompt_tokens_est_avg_per_block": round(statistics.mean(prompt_rows), 2) if prompt_rows else None,
        "output_tokens_est_total": sum(output_rows),
        "output_tokens_est_avg_per_block": round(statistics.mean(output_rows), 2) if output_rows else None,
        "duration_sec_total": round(sum(duration_rows), 2),
        "duration_sec_avg_per_block": round(statistics.mean(duration_rows), 3) if duration_rows else None,
    }


def _run_model(
    *,
    db_path: Path,
    out_dir: Path,
    model: str,
    reasoning: str,
    codex_bin: str,
    selection: list[dict[str, Any]],
) -> dict[str, Any]:
    model_dir = out_dir / model.replace(".", "_")
    model_dir.mkdir(parents=True, exist_ok=True)
    results_path = model_dir / "results.jsonl"
    progress_path = model_dir / "progress.json"
    summary_path = model_dir / "summary.json"

    completed = _load_completed(results_path)
    rows: list[dict[str, Any]] = []
    if results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    total_blocks = sum(int(item["block_count"]) for item in selection)
    processed_blocks = len(rows)

    for record_index, item in enumerate(selection, start=1):
        call_id = int(item["call_id"])
        source_filename, blocks = _fetch_record_blocks(db_path, call_id)
        for block in blocks:
            slot = str(block["slot"])
            if (call_id, slot) in completed:
                continue
            prompt = str(block["prompt"])
            with tempfile.NamedTemporaryFile(prefix="merge_bench_", suffix=".txt") as out_file:
                cmd = [
                    codex_bin,
                    "exec",
                    "--skip-git-repo-check",
                    "--ephemeral",
                    "--sandbox",
                    "read-only",
                    "--model",
                    model,
                    "--output-last-message",
                    out_file.name,
                    "-c",
                    f'model_reasoning_effort="{reasoning}"',
                ]
                append_codex_service_tier(cmd)
                cmd.append(prompt)
                started = time.time()
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=180,
                )
                elapsed = time.time() - started
                raw = Path(out_file.name).read_text(encoding="utf-8", errors="ignore").strip()

            payload: dict[str, Any] = {}
            parse_error = None
            if proc.returncode == 0:
                try:
                    payload = _extract_json(raw)
                except Exception as exc:  # noqa: BLE001
                    parse_error = str(exc)

            row = {
                "record_index": record_index,
                "call_id": call_id,
                "source_filename": source_filename,
                "slot": slot,
                "speaker_label": block["speaker_label"],
                "prompt_tokens_est": int(block["prompt_tokens_est"]),
                "tokens_used_actual": _parse_tokens_used(proc.stderr or ""),
                "output_tokens_est": _tok_count(raw) if raw else 0,
                "duration_sec": round(elapsed, 3),
                "returncode": int(proc.returncode),
                "parse_error": parse_error,
                "stderr_tail": (proc.stderr or "").strip().splitlines()[-6:],
                "selection": str(payload.get("selection") or ""),
                "confidence": payload.get("confidence"),
                "merged_text_chars": len(str(payload.get("merged_text") or "").strip()),
            }
            rows.append(row)
            _append_jsonl(results_path, row)
            processed_blocks += 1
            _write_progress(
                progress_path,
                {
                    "model": model,
                    "processed_blocks": processed_blocks,
                    "total_blocks": total_blocks,
                    "processed_records_hint": record_index,
                    "sample_size_records": len(selection),
                    "last_call_id": call_id,
                    "last_slot": slot,
                    "last_tokens_used_actual": row["tokens_used_actual"],
                    "last_duration_sec": row["duration_sec"],
                },
            )

    summary = _build_summary(model, rows, sample_records=len(selection))
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark codex_cli merge on existing dual-ASR variants")
    parser.add_argument("--db", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--models", default="gpt-5.4-mini,gpt-5.4")
    parser.add_argument("--reasoning", default="medium")
    parser.add_argument("--codex-bin", default="/Applications/Codex.app/Contents/Resources/codex")
    args = parser.parse_args()

    db_path = Path(args.db)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    selection = _load_selection(db_path, out_dir / "selection.json", args.sample_size)
    model_names = [item.strip() for item in str(args.models).split(",") if item.strip()]

    summaries: list[dict[str, Any]] = []
    for model in model_names:
        print(f"[benchmark] start model={model}", flush=True)
        summary = _run_model(
            db_path=db_path,
            out_dir=out_dir,
            model=model,
            reasoning=args.reasoning,
            codex_bin=args.codex_bin,
            selection=selection,
        )
        summaries.append(summary)
        print(f"[benchmark] done model={model} summary={json.dumps(summary, ensure_ascii=False)}", flush=True)

    comparison = {
        "db_path": str(db_path),
        "reasoning": args.reasoning,
        "sample_size_records": len(selection),
        "sample_size_blocks": sum(int(item["block_count"]) for item in selection),
        "summaries": summaries,
    }
    (out_dir / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(comparison, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
