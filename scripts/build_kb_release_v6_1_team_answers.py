#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from scripts import build_kb_release_v3_from_claude_handoff as kb_builder
from scripts.build_kb_distribution_packs import build_distribution_packs
from scripts.run_kb_semantic_review import run_kb_semantic_review


DEFAULT_SOURCE_ROOT = Path("/Users/dmitrijfabarisov/Claude Projects/Foton/kb_for_bot_review_2026-05-18")
DEFAULT_RUN_ID = "kb_release_20260520_v6_3_team_answers"
DEFAULT_SOURCE_OUT = Path("product_data/knowledge_base/kb_release_20260520_v6_3_team_answers_sources")
DEFAULT_RELEASE_OUT = Path("product_data/knowledge_base/kb_release_20260520_v6_3_team_answers")
DEFAULT_HANDOFF_OUT = Path("product_data/knowledge_base/kb_release_20260520_v6_3_team_answers_handoff_for_claude_and_team")
DEFAULT_BOT_PACK_OUT = Path("product_data/knowledge_base/kb_release_20260520_v6_3_team_answers_bot_pack")
DEFAULT_EMPLOYEE_PACK_OUT = Path("product_data/knowledge_base/kb_release_20260520_v6_3_team_answers_employee_pack")
DEFAULT_SMOKE_NOT_RUN = Path("product_data/knowledge_base/kb_release_20260520_v6_3_team_answers_smoke_not_run")


class SourceValidationError(ValueError):
    pass


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build KB v6.3 from YAML sources without live smoke runs.")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--source-out", type=Path, default=DEFAULT_SOURCE_OUT)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--release-out", type=Path, default=DEFAULT_RELEASE_OUT)
    parser.add_argument("--handoff-out", type=Path, default=DEFAULT_HANDOFF_OUT)
    parser.add_argument("--bot-pack-out", type=Path, default=DEFAULT_BOT_PACK_OUT)
    parser.add_argument("--employee-pack-out", type=Path, default=DEFAULT_EMPLOYEE_PACK_OUT)
    parser.add_argument("--smoke-dir", type=Path, default=DEFAULT_SMOKE_NOT_RUN)
    args = parser.parse_args(argv)

    source_out = prepare_source_overlay(args.source_root, args.source_out)
    manifest = load_release_manifest(source_out)
    validate_source_overlay(source_out, manifest)
    apply_release_manifest(manifest)

    build_result = kb_builder.build_kb_release_v3(
        run_id=args.run_id,
        handoff_dir=source_out,
        out_dir=args.release_out,
        handoff_out_dir=args.handoff_out,
    )

    semantic_report = run_kb_semantic_review(args.handoff_out, out_dir=args.handoff_out)
    copy_if_exists(args.handoff_out / "semantic_review.json", args.release_out / "semantic_review.json")
    copy_if_exists(args.handoff_out / "semantic_review.md", args.release_out / "semantic_review.md")

    create_not_run_smoke_summaries(args.smoke_dir)
    pack_result = build_distribution_packs(
        release_dir=args.handoff_out,
        full_release_dir=args.release_out,
        smoke_dir=args.smoke_dir,
        employee_out=args.employee_pack_out,
        bot_out=args.bot_pack_out,
    )
    write_diff_summary(args.release_out, args.handoff_out)

    result = {
        "schema_version": "kb_release_v6_3_build_result_v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": args.run_id,
        "source_out": str(args.source_out),
        "release_out": str(args.release_out),
        "handoff_out": str(args.handoff_out),
        "bot_pack_out": str(args.bot_pack_out),
        "employee_pack_out": str(args.employee_pack_out),
        "source_manifest": str(source_out / "release_manifest.yaml"),
        "source_mutation_policy": "read_only_yaml_sources_no_business_patches_in_python",
        "smoke_status": "not_run_by_builder",
        "build_result": dict(build_result),
        "semantic_pass": bool(semantic_report.get("semantic_pass")),
        "semantic_blocking_findings": semantic_report.get("blocking_findings"),
        "pack_result": dict(pack_result),
    }
    (args.release_out / "v6_1_build_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if semantic_report.get("semantic_pass") else 2


def prepare_source_overlay(source_root: Path, source_out: Path) -> Path:
    source = source_root.expanduser().resolve(strict=False)
    target = source_out.expanduser().resolve(strict=False)
    if not source.exists():
        raise FileNotFoundError(source)
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)
    return target


def load_release_manifest(source_root: Path) -> dict[str, Any]:
    manifest_path = source_root / "release_manifest.yaml"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing {manifest_path}. v6.3 builder reads business overrides from release_manifest.yaml, not Python."
        )
    manifest = load_yaml(manifest_path)
    if not isinstance(manifest.get("control_numbers"), Mapping):
        raise SourceValidationError("release_manifest.yaml must define control_numbers")
    return manifest


def validate_source_overlay(source_root: Path, manifest: Mapping[str, Any]) -> None:
    required_files = manifest.get("required_source_files") or []
    missing = [rel for rel in required_files if not (source_root / str(rel)).exists()]
    if missing:
        raise SourceValidationError(f"Missing required source files: {missing}")

    yaml_cache: dict[Path, Mapping[str, Any]] = {}
    for check in manifest.get("required_yaml_paths") or []:
        if not isinstance(check, Mapping):
            raise SourceValidationError(f"Invalid required_yaml_paths item: {check!r}")
        rel = Path(str(check.get("file") or ""))
        dotted_path = str(check.get("path") or "")
        if not rel or not dotted_path:
            raise SourceValidationError(f"Invalid required_yaml_paths item: {check!r}")
        source_path = source_root / rel
        if source_path not in yaml_cache:
            yaml_cache[source_path] = load_yaml(source_path)
        if lookup_path(yaml_cache[source_path], dotted_path) is None:
            reason = check.get("reason") or "required by release manifest"
            raise SourceValidationError(f"Missing YAML path {rel}:{dotted_path} ({reason})")


def lookup_path(payload: Mapping[str, Any], dotted_path: str) -> Any:
    current: Any = payload
    for part in dotted_path.split("."):
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        else:
            return None
    return current


def apply_release_manifest(manifest: Mapping[str, Any]) -> None:
    control_numbers = as_mapping(manifest.get("control_numbers"))
    remove = {str(item) for item in as_sequence(control_numbers.get("remove"))}
    add = [str(item) for item in as_sequence(control_numbers.get("add"))]
    if remove:
        kb_builder.CONTROL_NUMBERS = tuple(number for number in kb_builder.CONTROL_NUMBERS if number not in remove)
    if add:
        kb_builder.CONTROL_NUMBERS = tuple(dict.fromkeys((*kb_builder.CONTROL_NUMBERS, *add)))

    builder_version = manifest.get("builder_version")
    if builder_version:
        kb_builder.BUILDER_VERSION = str(builder_version)
    freshness_check_date = manifest.get("freshness_check_date")
    if freshness_check_date:
        kb_builder.FRESHNESS_CHECK_DATE = str(freshness_check_date)

    for key, value in as_mapping(manifest.get("source_files")).items():
        if isinstance(value, Mapping):
            kb_builder.SOURCE_FILES.setdefault(str(key), dict(value))

    client_safe_path_markers = as_sequence(manifest.get("client_safe_path_markers"))
    if client_safe_path_markers:
        kb_builder.CLIENT_SAFE_PATH_MARKERS = tuple(str(item) for item in client_safe_path_markers if str(item or "").strip())

    manual_overrides = [
        dict(item)
        for item in as_sequence(manifest.get("manual_decision_fact_overrides"))
        if isinstance(item, Mapping)
    ]
    kb_builder.MANIFEST_MANUAL_DECISION_FACT_OVERRIDES = tuple(manual_overrides)

    structured_rules = [
        dict(item)
        for item in as_sequence(manifest.get("structured_metadata_rules"))
        if isinstance(item, Mapping)
    ]
    kb_builder.MANIFEST_STRUCTURED_METADATA_RULES = tuple(structured_rules)


def load_gold_answers_v3(source_root: Path | None = None) -> dict[str, Any]:
    root = source_root or DEFAULT_SOURCE_OUT
    return load_yaml(root / "facts" / "gold_answers_v3.yaml")


def gold_answers_v3_payload(source_root: Path | None = None) -> dict[str, Any]:
    """Backward-compatible test helper. The payload is read from YAML, not hardcoded."""
    return load_gold_answers_v3(source_root)


def create_not_run_smoke_summaries(root: Path) -> None:
    for brand in ("foton", "unpk"):
        target = root / brand
        target.mkdir(parents=True, exist_ok=True)
        (target / "stage6_eval_summary.json").write_text(
            json.dumps(
                {
                    "brand": brand,
                    "status": "not_run_by_builder",
                    "rows_total": 0,
                    "errors": 0,
                    "brand_separation_violation": 0,
                    "note": "The builder does not run Stage6. Real smoke results must be attached from an audit pack.",
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )


def write_diff_summary(release_out: Path, handoff_out: Path) -> None:
    text = """# DIFF v4 -> v6.3

## Source of truth

- Business facts are read from `facts/*.yaml`.
- Release metadata and control-number policy are read from `release_manifest.yaml`.
- The Python builder validates source paths and assembles artifacts; it does not patch prices, contacts, transfers, brand rules, bot policy, or gold answers over YAML.

## Not run by builder

- Full MEGA smoke.
- Stage6/Codex smoke.
- Any live write to AMO/CRM/Tallanto.
"""
    (release_out / "DIFF_v4_vs_v6_1.md").write_text(text, encoding="utf-8")
    (handoff_out / "DIFF_v4_vs_v6_1.md").write_text(text, encoding="utf-8")


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping in {path}")
    return payload


def copy_if_exists(source: Path, target: Path) -> None:
    if source.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def as_sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    return []


if __name__ == "__main__":
    raise SystemExit(main())
