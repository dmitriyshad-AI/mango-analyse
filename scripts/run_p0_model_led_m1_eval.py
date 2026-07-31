#!/usr/bin/env python3
"""Measure the existing direct-path P0 model field with exactly one LLM call per case."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

from mango_mvp.channels.p0_recall_spec import hard_codes_from_text
from mango_mvp.channels.subscription_llm_parts.direct_path import _build_direct_path_prompt
from mango_mvp.channels.subscription_llm_parts.provider import SubscriptionLlmDraftProvider
from mango_mvp.channels.subscription_llm_parts.support import (
    DIRECT_PATH_MODEL_P0_ENV,
    P0_MODEL_CLASSES_V2_ENV,
    P0_MODEL_LED_ENV,
)
from mango_mvp.insights.sanitizers import COMMON_SINGLE_NAME_RE
from mango_mvp.replay_exam.pseudonymizer import (
    PROGRAM_NAME_STOP_PHRASES,
    RU_MIXED_CASE_SURNAME_RE,
    RU_NAME_RE,
    pii_signals,
)


SCHEMA_VERSION = "p0_model_led_m1_eval_v1_2026_07_29"
VALID_LABELS = {"p0", "benign", "ambiguous"}
COMMON_SINGLE_NAME_ANY_CASE_RE = re.compile(COMMON_SINGLE_NAME_RE.pattern, re.I)
PATRONYMIC_RE = re.compile(r"\b[а-яё]{3,}(?:ович|евич|овна|евна|ична|инична)\b", re.I)
EXTRA_SINGLE_NAME_RE = re.compile(r"\b(?:л[её]ш(?:а|е|у|ей)?|юли(?:я|и|ю|ей))\b", re.I)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _case_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]


def _git_head() -> str:
    completed = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def load_cases(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        raw = json.loads(line)
        if not isinstance(raw, Mapping):
            raise ValueError(f"line {line_no}: row must be an object")
        text = str(raw.get("text") or "").strip()
        label = str(raw.get("p0_label") or raw.get("label") or "").strip().casefold()
        if not text or label not in VALID_LABELS:
            raise ValueError(f"line {line_no}: non-empty text and label p0|benign|ambiguous are required")
        recent_messages = tuple(
            str(item).strip()
            for item in (raw.get("recent_messages") or ())
            if str(item).strip()
        )[-6:]
        visible_text = "\n".join((text, *recent_messages))
        for phrase in PROGRAM_NAME_STOP_PHRASES:
            visible_text = visible_text.replace(phrase, "")
        name_found = bool(
            RU_NAME_RE.search(visible_text)
            or RU_MIXED_CASE_SURNAME_RE.search(visible_text)
            or COMMON_SINGLE_NAME_ANY_CASE_RE.search(visible_text)
            or PATRONYMIC_RE.search(visible_text)
            or EXTRA_SINGLE_NAME_RE.search(visible_text)
        )
        pii = sorted(set(pii_signals((text, *recent_messages))) | ({"person_name"} if name_found else set()))
        if pii:
            raise ValueError(f"line {line_no}: PII signals are forbidden in M1 input: {','.join(pii)}")
        case_id = str(raw.get("case_id") or _case_id(text)).strip()
        if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,120}", case_id):
            raise ValueError(f"line {line_no}: case_id must be an opaque safe identifier")
        rows.append(
            {
                "case_index": len(rows) + 1,
                "case_id": case_id,
                "text": text,
                "label": label,
                "class": str(raw.get("class") or "none").strip()[:80],
                "p0_classes": tuple(
                    dict.fromkeys(
                        str(item).strip()[:80]
                        for item in (raw.get("p0_classes") or ())
                        if str(item).strip()
                    )
                ),
                "source": str(raw.get("source") or "unknown").strip()[:80],
                "brand": str(raw.get("brand") or "unknown").strip().casefold()[:40],
                "review_status": str(raw.get("review_status") or "single_reviewer").strip()[:80],
                "expected_route": str(raw.get("expected_route") or "unspecified").strip()[:80],
                "recent_messages": recent_messages,
            }
        )
    if not rows:
        raise ValueError("empty P0 evaluation set")
    return rows


def evaluate_case(
    case: Mapping[str, Any],
    *,
    provider: SubscriptionLlmDraftProvider,
) -> dict[str, Any]:
    text = str(case["text"])
    context = {
        "active_brand": str(case.get("brand") or "unknown"),
        DIRECT_PATH_MODEL_P0_ENV: "1",
        P0_MODEL_CLASSES_V2_ENV: "1",
        P0_MODEL_LED_ENV: "1",
    }
    if case.get("recent_messages"):
        context["recent_messages"] = tuple(case["recent_messages"])
    prompt = _build_direct_path_prompt(text, context=context, facts={}, fact_pack={})
    result = provider._direct_path_draft_runner(prompt)
    metadata = result.metadata if isinstance(result.metadata, Mapping) else {}
    model = metadata.get("direct_path_model_p0")
    model = model if isinstance(model, Mapping) else {}
    regex_codes = tuple(hard_codes_from_text(text))
    return {
        "schema_version": SCHEMA_VERSION,
        "case_index": int(case["case_index"]),
        "case_id": str(case.get("case_id") or _case_id(text)),
        "label": str(case["label"]),
        "class": str(case["class"]),
        "p0_classes": list(case.get("p0_classes") or ()),
        "source": str(case["source"]),
        "review_status": str(case.get("review_status") or "single_reviewer"),
        "expected_route": str(case.get("expected_route") or "unspecified"),
        "model_field_present": (
            bool(model.get("is_p0_present"))
            if "is_p0_present" in model
            else "is_p0" in model
        ),
        "model_is_p0": bool(model.get("is_p0")),
        "model_p0_kind": str(model.get("p0_kind_raw") or model.get("p0_kind") or "")[:80],
        "regex_is_p0": bool(regex_codes),
        "regex_codes": list(regex_codes),
    }


def summarize(rows: Sequence[Mapping[str, Any]], *, denominator: int) -> dict[str, Any]:
    counters: Counter[str] = Counter()
    classes: dict[str, Counter[str]] = {}
    for row in rows:
        if row.get("label") == "ambiguous":
            counters["report_only_ambiguous"] += 1
            continue
        expected = row.get("label") == "p0"
        model = bool(row.get("model_is_p0"))
        regex = bool(row.get("regex_is_p0"))
        counters[f"model_{'tp' if expected and model else 'fn' if expected else 'fp' if model else 'tn'}"] += 1
        counters[f"regex_{'tp' if expected and regex else 'fn' if expected else 'fp' if regex else 'tn'}"] += 1
        if not row.get("model_field_present"):
            counters["model_field_missing"] += 1
        if row.get("source") == "traffic_hit" and not expected:
            counters["regex_false_positive_traffic_hits"] += 1
        row_classes = tuple(row.get("p0_classes") or ()) or (str(row.get("class") or "none"),)
        for class_name in row_classes:
            class_counter = classes.setdefault(str(class_name), Counter())
            class_counter["total"] += 1
            class_counter["model_correct"] += int(model == expected)
            class_counter["regex_correct"] += int(regex == expected)
    return {
        "schema_version": SCHEMA_VERSION,
        "rows": len(rows),
        "traffic_denominator": int(denominator),
        "counters": dict(sorted(counters.items())),
        "by_class": {key: dict(value) for key, value in sorted(classes.items())},
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-call P0 model evaluation for M1")
    parser.add_argument("--set", dest="set_path", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--parallel", type=int, default=3)
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--timeout-sec", type=int, default=180)
    parser.add_argument("--traffic-denominator", type=int, default=27_507)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cases = load_cases(args.set_path)
    if args.validate_only:
        print(json.dumps({"valid": True, "cases": len(cases), "set_sha256": _sha256(args.set_path)}))
        return 0
    if not 1 <= args.parallel <= 6:
        raise ValueError("--parallel must be 1..6")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.out_dir / "p0_model_results.jsonl"
    if results_path.exists():
        raise FileExistsError(f"refusing to overwrite {results_path}")

    def run(case: Mapping[str, Any]) -> dict[str, Any]:
        provider = SubscriptionLlmDraftProvider(
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            timeout_sec=args.timeout_sec,
            max_attempts=1,
            codex_isolated=True,
        )
        return evaluate_case(case, provider=provider)

    rows: list[dict[str, Any]] = []
    with results_path.open("a", encoding="utf-8") as output, ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {pool.submit(run, case): int(case["case_index"]) for case in cases}
        for future in as_completed(futures):
            try:
                row = future.result()
            except Exception as exc:  # noqa: BLE001 - one bad case must not erase the measured batch
                row = {"schema_version": SCHEMA_VERSION, "case_index": futures[future], "error_type": type(exc).__name__}
            rows.append(row)
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            output.flush()
    rows.sort(key=lambda row: int(row["case_index"]))
    summary = summarize([row for row in rows if "error_type" not in row], denominator=args.traffic_denominator)
    summary.update(
        {
            "code_commit": _git_head(),
            "input_cases": len(cases),
            "errors": sum("error_type" in row for row in rows),
            "llm_calls_expected": len(cases),
            "llm_calls_attempted": len(rows),
            "model": str(args.model),
            "reasoning_effort": str(args.reasoning_effort),
            "set_sha256": _sha256(args.set_path),
        }
    )
    summary_path = args.out_dir / "p0_model_summary.json"
    _atomic_json(summary_path, summary)
    _atomic_json(
        args.out_dir / "sha_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "code_commit": summary["code_commit"],
            "set_sha256": summary["set_sha256"],
            "results_sha256": _sha256(results_path),
            "summary_sha256": _sha256(summary_path),
        },
    )
    return 0 if not summary["errors"] and not summary["counters"].get("model_field_missing") else 3


if __name__ == "__main__":
    raise SystemExit(main())
