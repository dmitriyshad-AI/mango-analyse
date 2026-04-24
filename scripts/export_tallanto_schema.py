#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def load_env(path: str) -> None:
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key.strip()] = value.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Tallanto schema metadata for Mango analyse.")
    parser.add_argument("--env", help="Optional env file with CRM_TALLANTO_* values.")
    parser.add_argument("--output-dir", required=True, help="Directory for exported JSON files.")
    args = parser.parse_args()

    if args.env:
        load_env(args.env)

    from mango_mvp.amocrm_runtime.tallanto_api import TallantoApiClient, build_tallanto_api_config
    from mango_mvp.amocrm_runtime.tallanto_export import export_tallanto_schema_bundle

    client = TallantoApiClient(build_tallanto_api_config())
    written = export_tallanto_schema_bundle(client, output_dir=args.output_dir)
    print(json.dumps(written, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
