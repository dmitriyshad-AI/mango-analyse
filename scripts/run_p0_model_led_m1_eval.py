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
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from mango_mvp.channels.p0_recall_spec import hard_codes_from_text
from mango_mvp.channels.subscription_llm_parts.direct_path import DIRECT_SLOT_TOPIC_SHADOW_ENV, SEMANTIC_FRAME_POSTHOC_SHADOW_ENV, _build_direct_path_prompt
from mango_mvp.channels.subscription_llm_parts.policy_routing import AUTONOMOUS_ROUTES
from mango_mvp.channels.subscription_llm_parts.provider import SubscriptionLlmDraftProvider, _apply_direct_path_model_p0_route, _direct_path_p0_shadow_metadata
from mango_mvp.channels.subscription_llm_parts.support import DIRECT_PATH_ENV, DIRECT_PATH_MODEL_P0_ENV, DIRECT_PATH_PILOT_CONFIG_ENV, LLM_RETRIEVE_ENV
from mango_mvp.channels.subscription_llm_parts.support import P0_MODEL_CLASSES_V2_ENV, P0_MODEL_LED_ENV, ROUTE_RUBRIC_ENV, SEMANTIC_OUTPUT_VERIFIER_ENV
from mango_mvp.insights.sanitizers import COMMON_SINGLE_NAME_RE
from mango_mvp.replay_exam.pseudonymizer import (
    PROGRAM_NAME_STOP_PHRASES,
    RU_MIXED_CASE_SURNAME_RE,
    RU_NAME_RE,
    pii_signals,
)


SCHEMA_VERSION = "p0_model_led_m1_eval_v1_2026_07_29"
VALID_LABELS = {"p0", "benign", "ambiguous"}
TRAFFIC_CORPUS_DENOMINATOR = 27_507
EXPECTED_LABEL_COUNTS = {"p0": 298, "benign": 496, "ambiguous": 21}
EXPECTED_SET_SHA256 = "00067d63473cbb6000311f1828e0845c638001ee4d61935ad45308dba7c24450"
MAX_MODEL_FALSE_POSITIVES = 10
SAFE_METADATA_VALUES = {
    "brand": {"foton", "unpk", "unknown"},
    "class": {"child_safety", "complaint", "legal", "none", "payment_dispute", "refund"},
    "p0_classes": {"child_safety", "complaint", "legal", "none", "payment_dispute", "refund"},
    "source": {"paraphrase", "traffic_hit", "traffic_miss", "traffic_prior_audit_candidate", "unknown"},
    "review_status": {"dual_blind_agreement", "dual_blind_disagreement", "hidden_context_missing_from_model_input", "independent_full_set_review_pass", "needs_context", "owner_architect_context_adjudication", "single_reviewer"},
    "expected_route": {"manager_only", "manual_review", "non_p0_unspecified", "unspecified"},
}
COMMON_SINGLE_NAME_ANY_CASE_RE = re.compile(COMMON_SINGLE_NAME_RE.pattern, re.I)
PATRONYMIC_RE = re.compile(r"\b[а-яё]{3,}(?:ович|евич|овна|евна|ична|инична)\b", re.I)
EXTRA_SINGLE_NAME_RE = re.compile(r"\b(?:л[её]ш(?:а|е|у|ей)?|юли(?:я|и|ю|ей))\b", re.I)


class _ReplayProvider(SubscriptionLlmDraftProvider):
    def __init__(self, result: Any) -> None:
        self.external_calls = 0
        super().__init__(runner=self._forbid_external_call, max_attempts=1, codex_isolated=True)
        self.result = result
        self.calls = 0

    def _forbid_external_call(self, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        self.external_calls += 1
        raise RuntimeError("external model call forbidden during deterministic replay")
    def _direct_path_draft_runner(self, prompt: str) -> Any:
        del prompt
        self.calls += 1
        return self.result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _case_id(text: str) -> str:
    return "synthetic_" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]


def _git_head() -> str:
    status = subprocess.run(
        ("git", "status", "--porcelain", "--untracked-files=all"), cwd=Path(__file__).resolve().parents[1], check=True, capture_output=True, text=True
    )
    if status.stdout.strip():
        raise RuntimeError("M1 evaluation requires a clean git worktree")
    completed = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _safe_metadata_token(value: Any, *, default: str, field: str, line_no: int) -> str:
    token = str(value or default).strip()
    if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,80}", token) or token not in SAFE_METADATA_VALUES[field]:
        raise ValueError(f"line {line_no}: {field} must be an opaque safe token")
    return token


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
        case_id = _case_id(str(raw.get("case_id") or text))
        rows.append(
            {
                "case_index": len(rows) + 1,
                "case_id": case_id,
                "text": text,
                "label": label,
                "class": _safe_metadata_token(raw.get("class"), default="none", field="class", line_no=line_no),
                "p0_classes": tuple(
                    dict.fromkeys(
                        _safe_metadata_token(item, default="none", field="p0_classes", line_no=line_no)
                        for item in (raw.get("p0_classes") or ())
                        if str(item).strip()
                    )
                ),
                "source": _safe_metadata_token(raw.get("source"), default="unknown", field="source", line_no=line_no),
                "brand": _safe_metadata_token(raw.get("brand"), default="unknown", field="brand", line_no=line_no).casefold(),
                "review_status": _safe_metadata_token(
                    raw.get("review_status"), default="single_reviewer", field="review_status", line_no=line_no
                ),
                "expected_route": _safe_metadata_token(
                    raw.get("expected_route"), default="unspecified", field="expected_route", line_no=line_no
                ),
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
    shadow = _direct_path_p0_shadow_metadata(result, client_message=text, context=context)
    regex_codes = tuple(shadow.get("regex_codes") or hard_codes_from_text(text))
    probe_input = replace(result, route="bot_answer_self_for_pilot", message_type="answer", draft_text="Проверка модельного P0-сигнала.")
    signal_probe = _apply_direct_path_model_p0_route(probe_input, client_message="Уточняю вопрос.", context=context)
    replay_context = {
        **context,
        DIRECT_PATH_ENV: "1",
        DIRECT_PATH_PILOT_CONFIG_ENV: "p0_m1_deterministic_replay",
        **{flag: "0" for flag in (DIRECT_SLOT_TOPIC_SHADOW_ENV, SEMANTIC_FRAME_POSTHOC_SHADOW_ENV, SEMANTIC_OUTPUT_VERIFIER_ENV, LLM_RETRIEVE_ENV, ROUTE_RUBRIC_ENV)},
    }
    model_replay = _ReplayProvider(result)
    legacy_replay = _ReplayProvider(result)
    model_led_result = model_replay.build_draft(text, context=replay_context)
    legacy_context = {**replay_context, DIRECT_PATH_MODEL_P0_ENV: "0", P0_MODEL_CLASSES_V2_ENV: "0", P0_MODEL_LED_ENV: "0"}
    legacy_result = legacy_replay.build_draft(text, context=legacy_context)
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
        "model_field_present": bool(shadow.get("model_field_present")),
        "model_field_valid": bool(shadow.get("model_field_valid")),
        "model_contract_status": str(shadow.get("model_contract_status") or "missing"),
        "model_is_p0": bool(model.get("is_p0")),
        "model_effective_is_p0": bool(shadow.get("model_effective_is_p0")),
        "model_p0_kind": str(shadow.get("model_p0_kind") or "")[:80],
        "model_signal_route": signal_probe.route,
        "model_led_route": model_led_result.route,
        "legacy_route": legacy_result.route,
        "model_led_replay_calls": model_replay.calls,
        "legacy_replay_calls": legacy_replay.calls,
        "model_led_external_calls": model_replay.external_calls,
        "legacy_external_calls": legacy_replay.external_calls,
        "regex_is_p0": bool(regex_codes),
        "regex_codes": list(regex_codes),
    }


def summarize(rows: Sequence[Mapping[str, Any]], *, denominator: int) -> dict[str, Any]:
    counters: Counter[str] = Counter()
    classes: dict[str, Counter[str]] = {}
    label_counts = Counter(str(row.get("label") or "") for row in rows)
    for row in rows:
        if not row.get("model_field_present"):
            counters["model_field_missing"] += 1
        elif not row.get("model_field_valid", True):
            counters["model_field_invalid"] += 1
        replay_external_calls = int(row.get("model_led_external_calls") or 0) + int(row.get("legacy_external_calls") or 0)
        if replay_external_calls:
            counters["replay_external_calls"] += replay_external_calls
        for side in ("model_led", "legacy"):
            key = f"{side}_replay_calls"
            if key in row:
                calls = int(row.get(key) or 0)
                counters[f"{side}_replay_{'preblocked' if calls == 0 else 'one' if calls == 1 else 'invalid'}"] += 1
                counters["replay_call_invalid"] += int(calls not in (0, 1))
        if row.get("label") == "ambiguous":
            counters["report_only_ambiguous"] += 1
            continue
        expected = row.get("label") == "p0"
        model = bool(row.get("model_is_p0"))
        regex = bool(row.get("regex_is_p0"))
        model_led_route = str(row.get("model_led_route") or "")
        legacy_route = str(row.get("legacy_route") or "")
        model_signal_route = str(row.get("model_signal_route") or "")
        counters[f"model_{'tp' if expected and model else 'fn' if expected else 'fp' if model else 'tn'}"] += 1
        counters[f"regex_{'tp' if expected and regex else 'fn' if expected else 'fp' if regex else 'tn'}"] += 1
        if model_led_route and legacy_route:
            counters["route_pair_rows"] += 1
            counters["route_pair_changed"] += int(model_led_route != legacy_route)
        if expected and model_signal_route != "manager_only":
            counters["model_signal_p0_route_miss"] += 1
        if expected and model_led_route in AUTONOMOUS_ROUTES:
            counters["model_led_p0_autonomous_route"] += 1
        if row.get("source") == "traffic_hit" and not expected and regex:
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
        "source_corpus_denominator": int(denominator),
        "classification_denominator": sum(label_counts[label] for label in ("p0", "benign")),
        "ambiguous_denominator": label_counts["ambiguous"],
        "label_counts": dict(sorted(label_counts.items())),
        "counters": dict(sorted(counters.items())),
        "by_class": {key: dict(value) for key, value in sorted(classes.items())},
    }


def diagnostic_quality_passed(summary: Mapping[str, Any], *, errors: int) -> bool:
    counters = summary.get("counters") if isinstance(summary.get("counters"), Mapping) else {}
    return bool(
        errors == 0
        and summary.get("label_counts") == EXPECTED_LABEL_COUNTS
        and not counters.get("model_field_missing")
        and not counters.get("model_field_invalid")
        and not counters.get("model_fn")
        and not counters.get("model_signal_p0_route_miss")
        and not counters.get("model_led_p0_autonomous_route")
        and not counters.get("replay_external_calls")
        and not counters.get("replay_call_invalid") and counters.get("model_led_replay_one") and counters.get("legacy_replay_one")
        and counters.get("route_pair_rows") == summary.get("classification_denominator")
        and int(counters.get("model_fp") or 0) <= MAX_MODEL_FALSE_POSITIVES
    )


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
    parser.add_argument("--traffic-denominator", type=int, choices=(TRAFFIC_CORPUS_DENOMINATOR,), default=TRAFFIC_CORPUS_DENOMINATOR)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    set_sha256 = _sha256(args.set_path)
    if set_sha256 != EXPECTED_SET_SHA256:
        raise ValueError(f"unexpected M1 set sha256: {set_sha256}")
    code_commit = _git_head()
    cases = load_cases(args.set_path)
    if Counter(str(case["label"]) for case in cases) != Counter(EXPECTED_LABEL_COUNTS):
        raise ValueError(f"M1 set must have label counts {EXPECTED_LABEL_COUNTS}")
    if args.validate_only:
        print(json.dumps({"valid": True, "cases": len(cases), "set_sha256": set_sha256, "code_commit": code_commit}))
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
    errors = sum("error_type" in row for row in rows)
    summary = summarize([row for row in rows if "error_type" not in row], denominator=args.traffic_denominator)
    quality_passed = diagnostic_quality_passed(summary, errors=errors)
    summary.update(
        {
            "code_commit": code_commit,
            "input_cases": len(cases),
            "errors": errors,
            "llm_calls_expected": len(cases),
            "llm_calls_attempted": len(rows),
            "model": str(args.model),
            "reasoning_effort": str(args.reasoning_effort),
            "set_sha256": set_sha256,
            "evaluation_scope": "one_call_classification_and_paired_build_draft_replay",
            "quality_passed": quality_passed,
            "activation_ready": False,
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
    return 0 if quality_passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
