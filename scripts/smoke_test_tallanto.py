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
    parser = argparse.ArgumentParser(description="Tallanto smoke-check for Mango analyse runtime.")
    parser.add_argument("--env", help="Optional env file with CRM_TALLANTO_* values.")
    args = parser.parse_args()

    if args.env:
        load_env(args.env)

    from mango_mvp.amocrm_runtime.tallanto_api import TallantoApiClient, build_tallanto_api_config

    client = TallantoApiClient(build_tallanto_api_config())
    modules = client.list_possible_modules()
    contact_fields = client.list_possible_fields("Contact")
    opportunity_fields = client.list_possible_fields("Opportunity")

    payload = {
        "ok": True,
        "base_url": client.config.base_url,
        "module_count": len(modules) if isinstance(modules, dict) else None,
        "contact_field_count": len(contact_fields) if isinstance(contact_fields, dict) else None,
        "opportunity_field_count": len(opportunity_fields) if isinstance(opportunity_fields, dict) else None,
        "modules": list(modules.keys())[:20] if isinstance(modules, dict) else modules,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
