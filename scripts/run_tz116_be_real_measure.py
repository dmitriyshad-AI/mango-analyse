#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from mango_mvp.customer_timeline.canonical_readonly_import import infer_brand
from mango_mvp.insights.outcome_linker import SignalSummary, classify_tallanto_rows
from mango_mvp.insights.phone_identity import normalize_phone, phones_from_text


DEFAULT_MAIN_ROOT = Path("/Users/dmitrijfabarisov/Projects/Mango analyse")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tallanto_path = Path(args.tallanto_contacts).expanduser().resolve()
    client_chains_path = Path(args.client_chains).expanduser().resolve()
    master_contacts_path = Path(args.master_contacts).expanduser().resolve()
    amo_contacts_path = Path(args.amo_contacts).expanduser().resolve()
    amo_deals_path = Path(args.amo_deals).expanduser().resolve()

    tallanto_rows = read_csv(tallanto_path)
    tallanto_index = build_tallanto_shadow_index(tallanto_rows)
    b_summary, b_rows = measure_outcome_shadow(client_chains_path, tallanto_index)
    e_sources: list[dict[str, Any]] = []
    e_change_rows: list[dict[str, Any]] = []
    for label, path in (
        ("master_contacts", master_contacts_path),
        ("amo_contacts", amo_contacts_path),
        ("amo_deals", amo_deals_path),
    ):
        source_summary, source_changes = measure_brand_source(label, path)
        e_sources.append(source_summary)
        e_change_rows.extend(source_changes)

    summary = {
        "schema_version": "tz116_be_real_measure_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "read_only_measure",
        "inputs": {
            "tallanto_contacts": str(tallanto_path),
            "client_chains": str(client_chains_path),
            "master_contacts": str(master_contacts_path),
            "amo_contacts": str(amo_contacts_path),
            "amo_deals": str(amo_deals_path),
        },
        "b_outcome_negation_shadow": b_summary,
        "e_brand_infer": {
            "sources": e_sources,
            "total_changed_rows": sum(int(item["changed_rows"]) for item in e_sources),
        },
        "llm_calls_total": 0,
        "safety": {
            "calls_model": False,
            "writes_db": False,
            "writes_crm": False,
            "writes_tallanto": False,
            "copies_raw_pii_to_git": False,
        },
    }
    write_csv(out_dir / "outcome_shadow_changed_rows.csv", b_rows)
    write_csv(out_dir / "brand_infer_changed_rows.csv", e_change_rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "REPORT.md").write_text(render_report(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def build_tallanto_shadow_index(rows: Sequence[Mapping[str, Any]]) -> dict[str, SignalSummary]:
    rows_by_phone: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        phones: list[str] = []
        for key in ("phone_parent", "phone_extra", "phones_joined"):
            phones.extend(phones_from_text(row.get(key)))
        for phone in unique(phones):
            rows_by_phone[phone].append(dict(row))
    return {phone: classify_tallanto_rows(group, outcome_model_mode="shadow") for phone, group in rows_by_phone.items()}


def measure_outcome_shadow(
    client_chains_path: Path,
    tallanto_index: Mapping[str, SignalSummary],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    chain_rows = read_csv(client_chains_path)
    tallanto_changed = [
        signal for signal in tallanto_index.values() if (signal.metadata.get("outcome_model_shadow") or {}).get("label_changed")
    ]
    tallanto_flips = Counter(
        flip_key(signal.metadata.get("outcome_model_shadow") or {}) for signal in tallanto_changed
    )

    changed_rows: list[dict[str, Any]] = []
    chain_flips: Counter[str] = Counter()
    matched = 0
    changed = 0
    for row_index, row in enumerate(chain_rows, start=2):
        phone = normalize_phone(row.get("phone") or row.get("client_key"))
        signal = tallanto_index.get(phone or "")
        if signal is None:
            continue
        matched += 1
        shadow = signal.metadata.get("outcome_model_shadow") or {}
        if shadow.get("label_changed"):
            changed += 1
            key = flip_key(shadow)
            chain_flips[key] += 1
            changed_rows.append(
                {
                    "row_index": row_index,
                    "legacy_label": shadow.get("legacy_label", ""),
                    "semantic_label": shadow.get("semantic_label", ""),
                    "flip": key,
                    "semantic_reasons": " | ".join(shadow.get("semantic_reasons") or []),
                }
            )

    summary = {
        "tallanto_phone_index_total": len(tallanto_index),
        "tallanto_phone_index_changed": len(tallanto_changed),
        "tallanto_phone_index_flips": dict(tallanto_flips.most_common()),
        "client_chain_rows_total": len(chain_rows),
        "client_chain_rows_matched_to_tallanto": matched,
        "client_chain_rows_changed": changed,
        "client_chain_flips": dict(chain_flips.most_common()),
        "interpretation": "shadow disagreement / candidate fixes, not gold accuracy",
    }
    return summary, changed_rows


def measure_brand_source(label: str, path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = read_csv(path)
    legacy_counts: Counter[str] = Counter()
    v2_counts: Counter[str] = Counter()
    flips: Counter[str] = Counter()
    changes: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows, start=2):
        values = list(row.values())
        legacy = infer_brand(values, mode="legacy")
        v2 = infer_brand(values, mode="cyrillic_v2")
        legacy_counts[legacy] += 1
        v2_counts[v2] += 1
        if legacy != v2:
            key = f"{legacy}->{v2}"
            flips[key] += 1
            changes.append({"source": label, "row_index": row_index, "legacy": legacy, "cyrillic_v2": v2, "flip": key})
    summary = {
        "source": label,
        "path": str(path),
        "rows_total": len(rows),
        "legacy_known": sum(count for brand, count in legacy_counts.items() if brand != "unknown"),
        "cyrillic_v2_known": sum(count for brand, count in v2_counts.items() if brand != "unknown"),
        "known_delta": sum(count for brand, count in v2_counts.items() if brand != "unknown")
        - sum(count for brand, count in legacy_counts.items() if brand != "unknown"),
        "changed_rows": len(changes),
        "legacy_counts": dict(legacy_counts.most_common()),
        "cyrillic_v2_counts": dict(v2_counts.most_common()),
        "flips": dict(flips.most_common()),
    }
    return summary, changes


def flip_key(shadow: Mapping[str, Any]) -> str:
    return f"{shadow.get('legacy_label', '')}->{shadow.get('semantic_label', '')}"


def unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    fields = list(rows[0]) if rows else ["empty"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def render_report(summary: Mapping[str, Any]) -> str:
    b = summary["b_outcome_negation_shadow"]
    e = summary["e_brand_infer"]["sources"]
    lines = [
        "# TZ-116 B/E Real Measurement",
        "",
        "## B Outcome Negation Shadow",
        "",
        f"- Tallanto phones: `{b['tallanto_phone_index_total']}`",
        f"- Tallanto changed: `{b['tallanto_phone_index_changed']}`",
        f"- Client-chain matched: `{b['client_chain_rows_matched_to_tallanto']}`",
        f"- Client-chain changed: `{b['client_chain_rows_changed']}`",
        f"- Client-chain flips: `{json.dumps(b['client_chain_flips'], ensure_ascii=False, sort_keys=True)}`",
        "",
        "## E Brand Inference",
        "",
    ]
    for item in e:
        lines.append(
            f"- {item['source']}: rows `{item['rows_total']}`, legacy known `{item['legacy_known']}`, "
            f"v2 known `{item['cyrillic_v2_known']}`, delta `{item['known_delta']}`, changed `{item['changed_rows']}`"
        )
    lines.extend(
        [
            "",
            "Safety: read-only local CSV inputs, no model calls, no CRM/Tallanto writes, changed-row outputs do not include raw phone/name fields.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TZ-116 B/E: read-only real measurements for outcome negation and brand inference.")
    parser.add_argument("--out-dir", default="audits/_inbox/tz116_be_real_measure")
    parser.add_argument(
        "--tallanto-contacts",
        default=str(DEFAULT_MAIN_ROOT / "stable_runtime/tallanto_snapshot_20260331/tallanto_contacts_normalized.csv"),
    )
    parser.add_argument(
        "--client-chains",
        default=str(DEFAULT_MAIN_ROOT / "stable_runtime/insight_readiness_report_after_mango_update_20260521_v1/client_chains.csv"),
    )
    parser.add_argument(
        "--master-contacts",
        default=str(DEFAULT_MAIN_ROOT / "stable_runtime/sales_master_export_20260523_audio_working_store_v1/master_contacts_ru.csv"),
    )
    parser.add_argument(
        "--amo-contacts",
        default=str(DEFAULT_MAIN_ROOT / "stable_runtime/deal_aware_amo_live_snapshot_20260513_v2/amo_contacts_snapshot.csv"),
    )
    parser.add_argument(
        "--amo-deals",
        default=str(DEFAULT_MAIN_ROOT / "stable_runtime/deal_aware_amo_live_snapshot_20260513_v2/amo_deals_snapshot.csv"),
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
