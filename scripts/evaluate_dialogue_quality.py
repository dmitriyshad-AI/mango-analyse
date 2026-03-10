#!/usr/bin/env python3
"""Evaluate dialogue transcript quality signals from exported files."""

from __future__ import annotations

import argparse
import difflib
import json
import re
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


TIMED_LINE_RE = re.compile(
    r"^\[(?P<mm>\d{2}):(?P<ss>\d{2}(?:\.\d)?)\]\s+"
    r"(?P<speaker>Менеджер(?:\s*\([^)]+\))?|Клиент):\s*(?P<text>.*)$"
)
WORD_RE = re.compile(r"\S+", flags=re.UNICODE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate ordering and cross-talk quality for transcript exports."
    )
    parser.add_argument(
        "--transcripts-dir",
        required=True,
        help="Directory with exported *_text.txt and *_variants.json files.",
    )
    parser.add_argument(
        "--min-near-dup-chars",
        type=int,
        default=24,
        help="Minimum chars per line for near-duplicate cross-speaker check.",
    )
    parser.add_argument(
        "--near-dup-threshold",
        type=float,
        default=0.92,
        help="SequenceMatcher ratio threshold for residual near-duplicate detection.",
    )
    parser.add_argument(
        "--long-run-threshold",
        type=int,
        default=12,
        help="Max consecutive same-speaker lines considered suspicious.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional path to save JSON result. Prints to stdout regardless.",
    )
    return parser.parse_args()


def _safe_median(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.median(values))


def _safe_mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.mean(values))


def _timestamp_seconds(mm: str, ss: str) -> float:
    return int(mm) * 60.0 + float(ss)


def _list_files(root: Path, suffix: str) -> List[Path]:
    return sorted(root.rglob(f"*{suffix}"))


def _base_key(path: Path) -> str:
    name = path.name
    if name.endswith("_text.txt"):
        return name[: -len("_text.txt")]
    if name.endswith("_variants.json"):
        return name[: -len("_variants.json")]
    return path.stem


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _parse_timed_lines(path: Path) -> List[Tuple[float, str, str]]:
    rows: List[Tuple[float, str, str]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        match = TIMED_LINE_RE.match(raw.strip())
        if not match:
            continue
        ts = _timestamp_seconds(match.group("mm"), match.group("ss"))
        speaker_raw = match.group("speaker")
        speaker = "manager" if speaker_raw.startswith("Менеджер") else "client"
        text = match.group("text").strip()
        rows.append((ts, speaker, text))
    return rows


def evaluate_text_file(
    path: Path,
    min_near_dup_chars: int,
    near_dup_threshold: float,
) -> Dict[str, Any]:
    rows = _parse_timed_lines(path)
    backward_events = 0
    same_ts_cross_speaker_events = 0
    max_same_speaker_run = 0
    switches = 0
    near_dup_pairs = 0
    near_dup_max_ratio = 0.0
    prev_ts: Optional[float] = None
    prev_speaker: Optional[str] = None
    run_len = 0

    words = 0
    for idx, (ts, speaker, text) in enumerate(rows):
        words += len(WORD_RE.findall(text))
        if prev_ts is not None:
            if ts < prev_ts - 1e-6:
                backward_events += 1
            elif abs(ts - prev_ts) <= 1e-6 and prev_speaker is not None and prev_speaker != speaker:
                same_ts_cross_speaker_events += 1
        if prev_speaker == speaker:
            run_len += 1
        else:
            if prev_speaker is not None:
                switches += 1
            run_len = 1
        if run_len > max_same_speaker_run:
            max_same_speaker_run = run_len

        if idx > 0:
            p_ts, p_speaker, p_text = rows[idx - 1]
            if p_speaker != speaker:
                if len(p_text) >= min_near_dup_chars and len(text) >= min_near_dup_chars:
                    ratio = difflib.SequenceMatcher(None, p_text, text).ratio()
                    if ratio >= near_dup_threshold:
                        near_dup_pairs += 1
                        if ratio > near_dup_max_ratio:
                            near_dup_max_ratio = ratio
            _ = p_ts

        prev_ts = ts
        prev_speaker = speaker

    return {
        "lines": len(rows),
        "words": words,
        "switches": switches,
        "max_same_speaker_run": max_same_speaker_run,
        "backward_events": backward_events,
        "has_backward": backward_events > 0,
        "near_dup_pairs": near_dup_pairs,
        "near_dup_max_ratio": round(near_dup_max_ratio, 4),
        "same_ts_cross_speaker_events": same_ts_cross_speaker_events,
    }


def evaluate_variants(path: Path) -> Dict[str, Any]:
    payload = _load_json(path)
    warnings = payload.get("warnings")
    if not isinstance(warnings, list):
        warnings = []

    mode = str(payload.get("mode") or "")
    similarity_fallback = any(
        isinstance(w, str) and "channels_too_similar" in w for w in warnings
    )

    crosstalk = payload.get("stereo_crosstalk_dedupe")
    if not isinstance(crosstalk, dict):
        crosstalk = {}
    sequence_fix = payload.get("stereo_sequence_fix")
    if not isinstance(sequence_fix, dict):
        sequence_fix = {}
    time_fix = payload.get("stereo_time_fix")
    if not isinstance(time_fix, dict):
        time_fix = {}

    return {
        "mode": mode,
        "warnings": warnings,
        "warnings_count": len(warnings),
        "similarity_fallback": similarity_fallback,
        "stereo_dedupe_applied": bool(crosstalk.get("applied")),
        "stereo_dedupe_dropped_lines": int(crosstalk.get("dropped_lines") or 0),
        "stereo_sequence_fix_applied": bool(sequence_fix.get("applied")),
        "stereo_sequence_fix_swapped_pairs": int(
            sequence_fix.get("swapped_adjacent_pairs") or 0
        ),
        "stereo_time_fix_applied": bool(time_fix.get("applied")),
        "stereo_time_fix_adjusted_lines": int(
            time_fix.get("monotonic_adjusted_lines") or 0
        ),
    }


def main() -> int:
    args = parse_args()
    root = Path(args.transcripts_dir).expanduser().resolve()
    text_files = _list_files(root, "_text.txt")
    variants_files = _list_files(root, "_variants.json")

    text_by_key = {_base_key(p): p for p in text_files}
    variants_by_key = {_base_key(p): p for p in variants_files}
    all_keys = sorted(set(text_by_key.keys()) | set(variants_by_key.keys()))

    file_rows: List[Dict[str, Any]] = []
    warning_samples: List[Dict[str, Any]] = []
    backward_samples: List[Dict[str, Any]] = []
    long_run_samples: List[Dict[str, Any]] = []
    near_dup_file_rows: List[Dict[str, Any]] = []

    lines_values: List[float] = []
    words_values: List[float] = []
    switches_values: List[float] = []
    max_run_values: List[float] = []

    done = 0
    mode_stereo = 0
    mode_other = 0
    warnings_count = 0
    similarity_fallbacks = 0
    stereo_dedupe_applied = 0
    stereo_dedupe_dropped_lines_total = 0
    stereo_sequence_fix_applied = 0
    stereo_sequence_fix_swapped_pairs_total = 0
    stereo_time_fix_applied = 0
    stereo_time_fix_adjusted_lines_total = 0

    backward_timestamp_files = 0
    backward_timestamp_events = 0
    near_dup_files_count = 0
    near_dup_pairs_total = 0
    same_ts_cross_speaker_files = 0
    same_ts_cross_speaker_events_total = 0

    for key in all_keys:
        text_path = text_by_key.get(key)
        variants_path = variants_by_key.get(key)
        text_eval: Dict[str, Any] = {}
        variants_eval: Dict[str, Any] = {}

        if text_path is not None:
            text_eval = evaluate_text_file(
                text_path,
                min_near_dup_chars=args.min_near_dup_chars,
                near_dup_threshold=args.near_dup_threshold,
            )
            done += 1
            lines_values.append(float(text_eval["lines"]))
            words_values.append(float(text_eval["words"]))
            switches_values.append(float(text_eval["switches"]))
            max_run_values.append(float(text_eval["max_same_speaker_run"]))

            backward_timestamp_events += int(text_eval["backward_events"])
            if text_eval["has_backward"]:
                backward_timestamp_files += 1
                backward_samples.append(
                    {
                        "file": text_path.name,
                        "backward_events": text_eval["backward_events"],
                    }
                )

            same_ts_cross_speaker_events_total += int(
                text_eval["same_ts_cross_speaker_events"]
            )
            if int(text_eval["same_ts_cross_speaker_events"]) > 0:
                same_ts_cross_speaker_files += 1

            if int(text_eval["max_same_speaker_run"]) >= int(args.long_run_threshold):
                long_run_samples.append(
                    {
                        "file": text_path.name,
                        "max_same_speaker_run": text_eval["max_same_speaker_run"],
                    }
                )

            if int(text_eval["near_dup_pairs"]) > 0:
                near_dup_files_count += 1
                near_dup_pairs_total += int(text_eval["near_dup_pairs"])
                near_dup_file_rows.append(
                    {
                        "file": text_path.name,
                        "pairs": text_eval["near_dup_pairs"],
                        "max_similarity": text_eval["near_dup_max_ratio"],
                    }
                )

        if variants_path is not None:
            variants_eval = evaluate_variants(variants_path)
            if variants_eval["mode"] == "stereo":
                mode_stereo += 1
            else:
                mode_other += 1

            warnings_count += int(variants_eval["warnings_count"])
            if variants_eval["similarity_fallback"]:
                similarity_fallbacks += 1
            if variants_eval["stereo_dedupe_applied"]:
                stereo_dedupe_applied += 1
            stereo_dedupe_dropped_lines_total += int(
                variants_eval["stereo_dedupe_dropped_lines"]
            )
            if variants_eval["stereo_sequence_fix_applied"]:
                stereo_sequence_fix_applied += 1
            stereo_sequence_fix_swapped_pairs_total += int(
                variants_eval["stereo_sequence_fix_swapped_pairs"]
            )
            if variants_eval["stereo_time_fix_applied"]:
                stereo_time_fix_applied += 1
            stereo_time_fix_adjusted_lines_total += int(
                variants_eval["stereo_time_fix_adjusted_lines"]
            )

            if variants_eval["warnings"]:
                warning_samples.append(
                    {"file": variants_path.name, "warnings": variants_eval["warnings"]}
                )

        row = {
            "key": key,
            "text_file": text_path.name if text_path else None,
            "variants_file": variants_path.name if variants_path else None,
        }
        row.update(text_eval)
        row.update({f"v_{k}": v for k, v in variants_eval.items()})
        file_rows.append(row)

    near_dup_file_rows.sort(key=lambda x: (x["pairs"], x["max_similarity"]), reverse=True)
    backward_samples.sort(key=lambda x: x["backward_events"], reverse=True)
    long_run_samples.sort(key=lambda x: x["max_same_speaker_run"], reverse=True)

    result = {
        "summary": {
            "total_keys": len(all_keys),
            "text_files": len(text_files),
            "variants_files": len(variants_files),
            "done": done,
            "mode_stereo": mode_stereo,
            "mode_mono_or_fallback": mode_other,
            "warnings_count": warnings_count,
            "similarity_fallbacks": similarity_fallbacks,
            "stereo_dedupe_applied": stereo_dedupe_applied,
            "stereo_dedupe_dropped_lines_total": stereo_dedupe_dropped_lines_total,
            "stereo_sequence_fix_applied": stereo_sequence_fix_applied,
            "stereo_sequence_fix_swapped_pairs_total": stereo_sequence_fix_swapped_pairs_total,
            "stereo_time_fix_applied": stereo_time_fix_applied,
            "stereo_time_fix_adjusted_lines_total": stereo_time_fix_adjusted_lines_total,
            "backward_timestamp_files": backward_timestamp_files,
            "backward_timestamp_events": backward_timestamp_events,
            "same_ts_cross_speaker_files": same_ts_cross_speaker_files,
            "same_ts_cross_speaker_events": same_ts_cross_speaker_events_total,
            "residual_cross_speaker_near_duplicate_files": near_dup_files_count,
            "residual_cross_speaker_near_duplicate_pairs": near_dup_pairs_total,
        },
        "text_stats": {
            "avg_lines": round(_safe_mean(lines_values), 2),
            "median_lines": round(_safe_median(lines_values), 2),
            "avg_words": round(_safe_mean(words_values), 2),
            "median_words": round(_safe_median(words_values), 2),
            "avg_speaker_switches": round(_safe_mean(switches_values), 2),
            "median_speaker_switches": round(_safe_median(switches_values), 2),
            "avg_max_same_speaker_run": round(_safe_mean(max_run_values), 2),
            "median_max_same_speaker_run": round(_safe_median(max_run_values), 2),
            "max_same_speaker_run_overall": int(max(max_run_values) if max_run_values else 0),
        },
        "residual_cross_speaker_near_duplicates": {
            "files_count": near_dup_files_count,
            "pairs_total": near_dup_pairs_total,
            "top_files": near_dup_file_rows[:20],
        },
        "samples": {
            "warning_samples": warning_samples[:20],
            "backward_timestamp_samples": backward_samples[:20],
            "long_run_samples": long_run_samples[:20],
        },
        "config": {
            "transcripts_dir": str(root),
            "near_dup_threshold": args.near_dup_threshold,
            "min_near_dup_chars": args.min_near_dup_chars,
            "long_run_threshold": args.long_run_threshold,
        },
    }

    output = json.dumps(result, ensure_ascii=False, indent=2)
    print(output)
    if args.out:
        out_path = Path(args.out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
