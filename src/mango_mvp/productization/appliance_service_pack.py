from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional
from xml.sax.saxutils import escape as xml_escape

from mango_mvp.productization.product_db import guard_product_db_path
from mango_mvp.productization.test_ingest import path_is_relative_to


APPLIANCE_SERVICE_PACK_SCHEMA_VERSION = "appliance_service_pack_v1"


@dataclass(frozen=True)
class ApplianceServicePackSummary:
    schema_version: str
    product_root: str
    product_db_path: str
    out_dir: str
    templates_written: int
    installs_services: bool
    starts_services: bool
    validation_ok: bool
    blocked: int
    warnings: int

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


def build_appliance_service_pack(
    product_root: Path,
    product_db_path: Path,
    out_dir: Optional[Path] = None,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    python_bin: str = "python3",
) -> Mapping[str, Any]:
    product_root = product_root.resolve(strict=False)
    product_db_path = product_db_path.resolve(strict=False)
    out_dir = (out_dir or product_root / "service_pack").resolve(strict=False)
    guard_product_db_path(product_db_path, product_root, must_exist=False)
    guard_under_root(out_dir, product_root, "service pack output")
    if port < 1 or port > 65535:
        raise ValueError("port must be between 1 and 65535")
    out_dir.mkdir(parents=True, exist_ok=True)

    commands = service_commands(product_root, product_db_path, host=host, port=port, python_bin=python_bin)
    files = {
        out_dir / "launchd" / "com.mango-analyse.dashboard.plist": launchd_dashboard_plist(commands["serve_dashboard"]),
        out_dir / "systemd" / "mango-analyse-dashboard.service": systemd_service(
            name="Mango Analyse dashboard",
            command=commands["serve_dashboard"],
            working_dir=Path.cwd(),
        ),
        out_dir / "systemd" / "mango-analyse-scheduler-health.service": systemd_service(
            name="Mango Analyse scheduler health report",
            command=commands["scheduler_health"],
            working_dir=Path.cwd(),
        ),
        out_dir / "README.md": readme(commands),
    }
    written = []
    for path, content in files.items():
        guard_under_root(path, product_root, "service pack file")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        written.append({"path": str(path), "kind": path.suffix.lstrip(".") or path.name})
    report = {
        "summary": ApplianceServicePackSummary(
            schema_version=APPLIANCE_SERVICE_PACK_SCHEMA_VERSION,
            product_root=str(product_root),
            product_db_path=str(product_db_path),
            out_dir=str(out_dir),
            templates_written=len(written),
            installs_services=False,
            starts_services=False,
            validation_ok=True,
            blocked=0,
            warnings=0 if product_db_path.exists() else 1,
        ).to_json_dict(),
        "templates": written,
        "commands": commands,
        "manual_install_notes": [
            "Review generated templates before installing them.",
            "Set secrets via local environment files, not inside service templates.",
            "Run diagnostics before enabling a background service.",
        ],
        "safety": safety_contract(),
    }
    write_json(out_dir / "service_pack_manifest.json", report)
    return report


def service_commands(product_root: Path, product_db_path: Path, *, host: str, port: int, python_bin: str) -> Mapping[str, str]:
    return {
        "serve_dashboard": (
            f"PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src {python_bin} scripts/mango_office_product_api_http.py "
            f"--product-root {shell_path(product_root)} --product-db {shell_path(product_db_path)} "
            f"serve --host {host} --port {port}"
        ),
        "scheduler_health": (
            f"PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src {python_bin} scripts/mango_office_scheduler_health.py "
            f"--product-root {shell_path(product_root)} --product-db {shell_path(product_db_path)} "
            f"--out {shell_path(product_root / 'scheduler_health' / 'service_health.json')}"
        ),
        "diagnostics": (
            f"PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src {python_bin} scripts/mango_office_product_ops.py "
            f"--product-root {shell_path(product_root)} --product-db {shell_path(product_db_path)} diagnostics"
        ),
    }


def launchd_dashboard_plist(command: str) -> str:
    return """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.mango-analyse.dashboard</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/zsh</string>
    <string>-lc</string>
    <string>""" + xml_escape(command) + """</string>
  </array>
  <key>RunAtLoad</key>
  <false/>
  <key>KeepAlive</key>
  <false/>
</dict>
</plist>
"""


def systemd_service(*, name: str, command: str, working_dir: Path) -> str:
    escaped_command = command.replace('"', '\\"')
    return f"""[Unit]
Description={name}
After=network.target

[Service]
Type=simple
WorkingDirectory={working_dir}
ExecStart=/bin/sh -lc "{escaped_command}"
Restart=no

[Install]
WantedBy=multi-user.target
"""


def readme(commands: Mapping[str, str]) -> str:
    return f"""# Mango Analyse Appliance Service Pack

These files are templates only. They do not install or start services.

## Manual commands

```zsh
{commands['serve_dashboard']}
```

```zsh
{commands['scheduler_health']}
```

```zsh
{commands['diagnostics']}
```

Keep ASR execution, R+A execution, runtime DB writes and CRM writeback disabled
until the explicit acceptance gates are green.
"""


def shell_path(path: Path) -> str:
    text = str(path)
    if all(ch not in text for ch in (" ", "'", '"', "(", ")")):
        return text
    return "'" + text.replace("'", "'\"'\"'") + "'"


def guard_under_root(path: Path, product_root: Path, label: str) -> None:
    if "stable_runtime" in path.parts:
        raise ValueError(f"{label} must not be under stable_runtime")
    if not path_is_relative_to(path, product_root):
        raise ValueError(f"{label} must stay under product root: {product_root}")


def safety_contract() -> Mapping[str, bool]:
    return {
        "templates_only": True,
        "installs_services": False,
        "starts_services": False,
        "runtime_db_writes": False,
        "stable_runtime_writes": False,
        "downloads_audio": False,
        "run_asr": False,
        "run_ra": False,
        "write_crm": False,
        "write_tallanto": False,
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
