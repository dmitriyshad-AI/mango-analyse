#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


DEFAULT_WHATIF = Path(
    "/Users/dmitrijfabarisov/.mango_local/draft_loop_whatif/"
    "pair_missing_72h_latest25_20260630T080125Z/whatif_results.jsonl"
)
DEFAULT_OUT_DIR = Path("product_data/telegram_dynamic_test_sets")
DEFAULT_VERSION = "20260630"
DEFAULT_EXISTING_SETS = (
    "p0_model_led_micro_20260622.jsonl",
    "p0_stability_set_20260617.jsonl",
    "targeted_riskzones_2026_05_26.jsonl",
    "reliable_answerer_step1_20260625.jsonl",
    "closing_fix_tz142_20260627.jsonl",
    "p0_deep_match_tz147_20260618.jsonl",
)

PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-zА-Яа-я]{2,}")
LONG_ID_RE = re.compile(r"(?<!\d)\d{5,}(?!\d)")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            text = line.strip()
            if not text:
                continue
            item = json.loads(text)
            if isinstance(item, Mapping):
                rows.append(item)
    return rows


def _line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as file:
        return sum(1 for _ in file)


def _safe_text(value: Any, *, limit: int = 2400) -> str:
    text = str(value or "")
    text = EMAIL_RE.sub("[email]", text)
    text = PHONE_RE.sub("[phone]", text)
    text = LONG_ID_RE.sub("[id]", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > limit:
        return text[: limit - 1].rstrip() + "…"
    return text


def _safe_list(values: Iterable[Any], *, limit: int = 2000, max_items: int = 40) -> list[str]:
    output: list[str] = []
    for value in values:
        text = _safe_text(value, limit=limit)
        if text:
            output.append(text)
        if len(output) >= max_items:
            break
    return output


def _route_counts(rows: Iterable[Mapping[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts[str(row.get("bot_route") or "unknown")] += 1
    return counts


def _status_counts(rows: Iterable[Mapping[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts[str(row.get("status") or "unknown")] += 1
    return counts


def _flag_counts(rows: Iterable[Mapping[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        for flag in row.get("safety_flags") or []:
            if str(flag).strip():
                counts[str(flag)] += 1
    return counts


def _brand_counts(rows: Iterable[Mapping[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts[str(row.get("brand") or "unknown")] += 1
    return counts


def _case_from_whatif(row: Mapping[str, Any], *, ordinal: int) -> Mapping[str, Any]:
    source_idx = row.get("idx") if row.get("idx") is not None else ordinal
    return {
        "type": "wappi_whatif_case",
        "schema_version": "adr003_semantic_frame_eval_case_v1_2026_06_30",
        "case_id": f"wappi_pair_missing_72h_{ordinal:03d}",
        "source": {
            "kind": "wappi_pair_missing_72h_whatif",
            "source_event": str(row.get("source_event") or ""),
            "source_row_idx": source_idx,
            "message_time_msk": str(row.get("message_time_msk") or ""),
            "journal_created_at_msk": str(row.get("journal_created_at_msk") or ""),
        },
        "brand": str(row.get("brand") or ""),
        "channel": str(row.get("channel") or ""),
        "client_text": _safe_text(row.get("client_text"), limit=1200),
        "history_lines": _safe_list(row.get("history_lines") or [], limit=1800, max_items=30),
        "baseline": {
            "status": str(row.get("status") or ""),
            "bot_route": str(row.get("bot_route") or ""),
            "bot_text": _safe_text(row.get("bot_draft_text"), limit=5000),
            "safety_flags": [str(flag) for flag in (row.get("safety_flags") or []) if str(flag).strip()],
            "context_used": [str(item) for item in (row.get("context_used") or [])[:30] if str(item).strip()],
        },
        "acceptance_focus": [
            "route_text_noop_under_semantic_frame_shadow",
            "semantic_frame_observe_only",
            "manager_only_alignment_when_frame_must_handoff",
            "brand_and_fabrication_regression",
        ],
    }


def _existing_set_manifest(out_dir: Path) -> list[Mapping[str, Any]]:
    sources: list[Mapping[str, Any]] = []
    for name in DEFAULT_EXISTING_SETS:
        path = out_dir / name
        sources.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "line_count": _line_count(path) if path.exists() else 0,
                "sha256": _sha256(path) if path.exists() else "",
            }
        )
    return sources


def build_eval(*, whatif_path: Path, out_dir: Path, version: str) -> Mapping[str, Any]:
    if not whatif_path.exists():
        raise FileNotFoundError(f"what-if artifact not found: {whatif_path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_jsonl(whatif_path)
    drafted = [row for row in rows if str(row.get("status") or "") == "drafted"]
    selected = drafted[:25]
    if len(selected) != 25:
        raise RuntimeError(f"expected 25 drafted what-if rows, got {len(selected)}")

    cases = [_case_from_whatif(row, ordinal=index) for index, row in enumerate(selected, start=1)]
    eval_path = out_dir / f"adr003_semantic_frame_wappi_latest25_{version}.jsonl"
    with eval_path.open("w", encoding="utf-8") as file:
        for case in cases:
            file.write(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n")

    baseline = {
        "schema_version": "adr003_semantic_frame_baseline_v1_2026_06_30",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "whatif_source": str(whatif_path),
        "whatif_source_sha256": _sha256(whatif_path),
        "whatif_rows_total": len(rows),
        "whatif_drafted_rows": len(drafted),
        "wappi_latest25_selected": len(selected),
        "wappi_route_counts": dict(_route_counts(selected)),
        "wappi_status_counts": dict(_status_counts(selected)),
        "wappi_brand_counts": dict(_brand_counts(selected)),
        "wappi_safety_flag_counts": dict(_flag_counts(selected)),
        "existing_regression_sources": _existing_set_manifest(out_dir),
        "baseline_scope": {
            "source": "existing what-if drafts + frozen regression set manifest",
            "semantic_judge_status": "not_run_by_builder",
            "m1_repro_command_required": True,
            "notes": [
                "Builder freezes inputs and observed what-if route/text baseline.",
                "Bit-for-bit OFF/ON shadow diff and semantic gates must be measured by M1 run.",
                "No raw chat_id/message_id/phone/email is copied into the eval JSONL.",
            ],
        },
        "moratorium": "Новый провал понимания добавляется как eval-case; не добавлять новый детектор/SAFE_TEXT/флаг без ADR/review.",
    }
    baseline_path = out_dir / f"adr003_semantic_frame_baseline_{version}.json"
    baseline_path.write_text(json.dumps(baseline, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    manifest = {
        "schema_version": "adr003_semantic_frame_eval_manifest_v1_2026_06_30",
        "version": version,
        "generated_at": baseline["generated_at"],
        "wappi_latest25_path": str(eval_path),
        "baseline_path": str(baseline_path),
        "case_count": len(cases),
        "source_sets": [
            {
                "path": str(eval_path),
                "kind": "wappi_whatif_latest25_sanitized",
                "line_count": len(cases),
                "sha256": _sha256(eval_path),
            },
            *_existing_set_manifest(out_dir),
        ],
        "baseline_metrics": {
            "wappi_route_counts": baseline["wappi_route_counts"],
            "wappi_brand_counts": baseline["wappi_brand_counts"],
            "wappi_safety_flag_counts": baseline["wappi_safety_flag_counts"],
        },
        "m1_acceptance": {
            "shadow_off_on_route_text_diff": 0,
            "extra_model_calls_per_turn": 0,
            "frame_must_handoff_alignment_min": 0.95,
            "brand_leaks": 0,
            "fabrication_hard_gate_failures": 0,
        },
    }
    manifest_path = out_dir / f"adr003_semantic_frame_eval_manifest_{version}.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "eval_path": str(eval_path),
        "baseline_path": str(baseline_path),
        "manifest_path": str(manifest_path),
        "case_count": len(cases),
        "route_counts": baseline["wappi_route_counts"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ADR-003 SemanticFrame shadow eval manifest.")
    parser.add_argument("--whatif", type=Path, default=DEFAULT_WHATIF)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--version", default=DEFAULT_VERSION)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_eval(whatif_path=args.whatif, out_dir=args.out_dir, version=str(args.version))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
