#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
import re
from pathlib import Path
from typing import Any

from mango_mvp.channels.subscription_llm import build_codex_exec_command, build_codex_exec_env
from mango_mvp.question_catalog.codex_full_run import (
    CODEX_FULL_PROMPT_VERSION,
    CODEX_FULL_RUN_SCHEMA_VERSION,
    DEFAULT_FULL_RUN_OUT_ROOT,
    DEFAULT_TAXONOMY_PATH,
    CodexFullRunCandidate,
    batch_paths,
    build_batch_plan,
    build_full_classification_prompt,
    cache_key_for_batch,
    collect_prediction_rows,
    extract_json_object,
    guard_full_run_path,
    is_complete_batch,
    is_retryable_text,
    now_iso,
    read_question_item_rows,
    retry_delay,
    run_id_now,
    sha256_file,
    sha256_text,
    validate_and_normalize_predictions,
    write_batch_outputs,
    write_consolidated_outputs,
    write_json_atomic,
    write_text_atomic,
)


DEFAULT_INPUT = Path("product_data/question_catalog/customer_question_items.jsonl")


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    input_path = Path(args.input).resolve()
    taxonomy_path = Path(args.taxonomy).resolve()
    candidate = CodexFullRunCandidate.parse(args.candidate)
    run_id = args.run_id or run_id_now(f"{candidate.label}_{args.max_rows or 'full'}")
    out_root = guard_full_run_path(Path(args.out_root or DEFAULT_FULL_RUN_OUT_ROOT) / run_id, project_root=project_root)
    out_root.mkdir(parents=True, exist_ok=True)
    for name in ("raw", "predictions", "errors", "cache"):
        (out_root / name).mkdir(parents=True, exist_ok=True)

    rows = read_question_item_rows(input_path, max_rows=max(0, args.max_rows))
    batches = build_batch_plan(rows, batch_size=max(1, args.batch_size))
    taxonomy_sha = sha256_file(taxonomy_path)
    manifest = {
        "schema_version": CODEX_FULL_RUN_SCHEMA_VERSION,
        "prompt_version": CODEX_FULL_PROMPT_VERSION,
        "created_at": now_iso(),
        "run_id": run_id,
        "project_root": str(project_root),
        "input_path": str(input_path),
        "input_sha256": sha256_file(input_path),
        "taxonomy_path": str(taxonomy_path),
        "taxonomy_sha256": taxonomy_sha,
        "candidate": candidate.__dict__,
        "batch_size": max(1, args.batch_size),
        "max_rows": max(0, args.max_rows),
        "question_rows": len(rows),
        "batch_count": len(batches),
        "safety": {
            "codex_exec": True,
            "sandbox": "read-only",
            "uses_openai_api_key": False,
            "write_stable_runtime": False,
            "write_crm": False,
            "write_tallanto": False,
            "send_messages": False,
        },
    }
    write_json_atomic(out_root / "run_manifest.json", manifest)
    write_text_atomic(
        out_root / "batch_plan.jsonl",
        "".join(json.dumps({"batch_id": batch.batch_id, "item_ids": batch.item_ids}, ensure_ascii=False) + "\n" for batch in batches),
    )
    if args.dry_run_plan_only:
        result = finish_run(out_root, batches)
        print(json.dumps({"out_root": str(out_root), "dry_run_plan_only": True, **result}, ensure_ascii=False, indent=2))
        return

    errors: list[dict[str, Any]] = []
    for batch in batches:
        if args.resume and not args.force and is_complete_batch(out_root, batch):
            print(f"[skip] {batch.batch_id} already complete", flush=True)
            continue
        try:
            run_batch(
                batch,
                out_root=out_root,
                taxonomy_path=taxonomy_path,
                taxonomy_sha=taxonomy_sha,
                candidate=candidate,
                timeout_sec=max(30, args.timeout_sec),
                max_attempts=max(1, args.max_attempts),
                force=args.force,
                codex_bin=args.codex_bin,
            )
        except Exception as exc:  # noqa: BLE001
            error = {"batch_id": batch.batch_id, "error": str(exc)[:1000], "created_at": now_iso()}
            errors.append(error)
            write_json_atomic(out_root / "errors" / f"{batch.batch_id}.json", error)
            if not args.continue_on_error:
                raise
        finally:
            progress = build_progress(out_root, batches, errors=errors)
            write_json_atomic(out_root / "progress.json", progress)

    result = finish_run(out_root, batches)
    print(json.dumps({"out_root": str(out_root), **result}, ensure_ascii=False, indent=2))


def run_batch(
    batch,
    *,
    out_root: Path,
    taxonomy_path: Path,
    taxonomy_sha: str,
    candidate: CodexFullRunCandidate,
    timeout_sec: int,
    max_attempts: int,
    force: bool,
    codex_bin: str,
) -> None:
    prompt = build_full_classification_prompt(batch, taxonomy_path=taxonomy_path)
    prompt_sha = sha256_text(prompt)
    cache_key = cache_key_for_batch(prompt=prompt, batch=batch, candidate=candidate, taxonomy_sha256=taxonomy_sha)
    cache_path = out_root / "cache" / f"{cache_key}.json"
    raw_response = ""
    proc_stderr = ""
    tokens_used = 0
    elapsed = 0.0
    cache_hit = False
    if cache_path.exists() and not force:
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        raw_response = str(cache.get("raw_response") or "")
        proc_stderr = str(cache.get("stderr_tail") or "")
        tokens_used = int(cache.get("tokens_used") or 0)
        elapsed = float(cache.get("elapsed_sec") or 0.0)
        cache_hit = True
    else:
        for attempt in range(1, max_attempts + 1):
            started = time.time()
            try:
                raw_response, proc_stderr, tokens_used = call_codex_once(
                    prompt,
                    codex_bin=codex_bin,
                    candidate=candidate,
                    timeout_sec=timeout_sec,
                )
                elapsed = time.time() - started
                break
            except Exception as exc:  # noqa: BLE001
                message = str(exc)
                if attempt < max_attempts and is_retryable_text(message):
                    time.sleep(retry_delay(attempt))
                    continue
                raise
        write_json_atomic(
            cache_path,
            {
                "raw_response": raw_response,
                "stderr_tail": "\n".join(proc_stderr.splitlines()[-80:]),
                "tokens_used": tokens_used,
                "elapsed_sec": elapsed,
            },
        )

    response_sha = sha256_text(raw_response)
    payload = extract_json_object(raw_response)
    predictions = validate_and_normalize_predictions(
        payload,
        batch,
        candidate=candidate,
        taxonomy_sha256=taxonomy_sha,
        prompt_sha256=prompt_sha,
        response_sha256=response_sha,
    )
    write_batch_outputs(
        out_root,
        batch=batch,
        raw_response=raw_response,
        response_payload=payload,
        predictions=predictions,
        metadata={
            "schema_version": CODEX_FULL_RUN_SCHEMA_VERSION,
            "batch_id": batch.batch_id,
            "model": candidate.model,
            "reasoning_effort": candidate.reasoning_effort,
            "taxonomy_sha256": taxonomy_sha,
            "prompt_sha256": prompt_sha,
            "response_sha256": response_sha,
            "tokens_used": tokens_used,
            "elapsed_sec": elapsed,
            "cache_hit": cache_hit,
            "stderr_tail": "\n".join(proc_stderr.splitlines()[-80:]),
        },
    )
    paths = batch_paths(out_root, batch.batch_id)
    print(f"[done] {batch.batch_id} rows={len(predictions)} cache={cache_hit} path={paths['predictions_jsonl']}", flush=True)


def call_codex_once(
    prompt: str,
    *,
    codex_bin: str,
    candidate: CodexFullRunCandidate,
    timeout_sec: int,
) -> tuple[str, str, int]:
    with tempfile.NamedTemporaryFile(prefix="mango_qcatalog_codex_full_", suffix=".json") as out_file:
        output_path = Path(out_file.name)
        cmd = build_codex_exec_command(
            output_path=output_path,
            codex_bin=codex_bin,
            model=candidate.model,
            reasoning_effort=candidate.reasoning_effort,
        )
        proc = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_sec,
            env=build_codex_exec_env(),
        )
        raw = output_path.read_text(encoding="utf-8", errors="ignore")
    if proc.returncode != 0:
        stderr_tail = "\n".join((proc.stderr or "").splitlines()[-60:])
        raise RuntimeError(f"codex exec failed rc={proc.returncode}\n{stderr_tail}")
    return raw or proc.stdout or proc.stderr, proc.stderr or "", parse_tokens_used(proc.stderr or "")


def parse_tokens_used(stderr: str) -> int:
    match = re.search(r"tokens used\s*([0-9\s\u00a0]+)", stderr or "", flags=re.I)
    if not match:
        return 0
    digits = "".join(ch for ch in match.group(1) if ch.isdigit())
    return int(digits) if digits else 0


def finish_run(out_root: Path, batches) -> dict[str, Any]:
    rows = collect_prediction_rows(out_root, batches)
    outputs = write_consolidated_outputs(out_root, rows)
    progress = build_progress(out_root, batches, errors=[])
    write_json_atomic(out_root / "progress.json", progress)
    return {"complete_predictions": len(rows), "progress": progress, "outputs": outputs}


def build_progress(out_root: Path, batches, *, errors: list[dict[str, Any]]) -> dict[str, Any]:
    complete = sum(1 for batch in batches if is_complete_batch(out_root, batch))
    return {
        "schema_version": CODEX_FULL_RUN_SCHEMA_VERSION,
        "updated_at": now_iso(),
        "batch_count": len(batches),
        "complete_batches": complete,
        "pending_batches": max(0, len(batches) - complete),
        "errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run resumable Codex CLI classification over full question catalog.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--taxonomy", default=str(DEFAULT_TAXONOMY_PATH))
    parser.add_argument("--out-root", default=str(DEFAULT_FULL_RUN_OUT_ROOT))
    parser.add_argument("--run-id", default="")
    parser.add_argument("--candidate", default="gpt-5.5:xhigh", help="model:reasoning or label:model:reasoning")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--timeout-sec", type=int, default=480)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run-plan-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
