from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mango_mvp.knowledge_base.price_axes_catalog import build_price_axes_catalog


DEFAULT_SNAPSHOT = Path(
    "product_data/knowledge_base/kb_release_20260612_v6_7_staging_r4_1/kb_release_v3_snapshot.json"
)
DEFAULT_OUTPUT = DEFAULT_SNAPSHOT.with_name("price_axes_catalog.json")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build derived price-axis catalog from KB v3 snapshot.")
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    snapshot = _read_json(args.snapshot)
    catalog = build_price_axes_catalog(snapshot.get("facts") or [])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(catalog, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    entries = catalog.get("entries") or []
    issues = catalog.get("issues") or []
    print(f"snapshot={args.snapshot}")
    print(f"out={args.out}")
    print(f"entries={len(entries)}")
    print(f"issues={len(issues)}")
    print("unpk_online_entries=" + str(sum(1 for item in entries if item.get("source_kind") == "unpk_online_kc_source_price")))
    print("tariff_entries=" + str(sum(1 for item in entries if item.get("source_kind") == "foton_m9_m11_tariff_price")))
    print("range_issues=" + str(sum(1 for item in issues if item.get("issue") == "range_not_final_price")))
    print("empty_client_safe_issues=" + str(sum(1 for item in issues if item.get("issue") == "empty_client_safe_text_not_final_price")))
    return 0


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
