from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from mango_mvp.productization.tenant_config import load_tenant_config
from mango_mvp.productization.tenant_config_pinning import check_tenant_config_pin


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _valid_config() -> dict[str, object]:
    return {
        "schema_version": "tenant_config_v1",
        "tenant_id": "foton",
        "business": {"industry": "edtech"},
        "crm": {"target_fields": ["Авто история общения"], "protected_fields": ["Id Tallanto"]},
        "privacy": {"phone_in_ai_text": "redact"},
        "quality": {"crm_detector_min_severity": "P2"},
    }


def _write_config(path: Path, payload: dict[str, object] | None = None) -> None:
    path.write_text(
        json.dumps(payload or _valid_config(), ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_tenant_config_pin_passes_for_expected_hash(tmp_path: Path) -> None:
    path = tmp_path / "tenant_config_v1.json"
    _write_config(path)
    result = load_tenant_config(path)

    pin = check_tenant_config_pin(result, expected_sha256=result.sha256)

    assert pin["passed"] is True
    assert pin["reason"] == "ok"
    assert pin["actual_sha256"] == result.sha256
    assert pin["tenant_id"] == "foton"
    assert pin["schema_version"] == "tenant_config_v1"


def test_tenant_config_pin_fails_on_hash_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "tenant_config_v1.json"
    _write_config(path)
    result = load_tenant_config(path)

    pin = check_tenant_config_pin(result, expected_sha256="0" * 64)

    assert pin["passed"] is False
    assert pin["reason"] == "tenant_config_sha256_mismatch"


def test_tenant_config_pin_fails_when_config_not_loaded() -> None:
    pin = check_tenant_config_pin(None)

    assert pin["passed"] is False
    assert pin["reason"] == "tenant_config_not_loaded"


def test_tenant_config_pin_fails_on_tenant_id_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "tenant_config_v1.json"
    _write_config(path)
    result = load_tenant_config(path)

    pin = check_tenant_config_pin(result, expected_sha256=result.sha256, expected_tenant_id="other")

    assert pin["passed"] is False
    assert pin["reason"] == "tenant_config_tenant_id_mismatch"


def test_print_current_cli_outputs_hash_for_explicit_path(tmp_path: Path) -> None:
    path = tmp_path / "tenant_config_v1.json"
    _write_config(path)
    env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")}

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "mango_mvp.productization.tenant_config_pinning",
            "--print-current",
            "--path",
            str(path),
        ],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["path"] == str(path.resolve())
    assert payload["sha256"] == load_tenant_config(path).sha256
    assert payload["tenant_id"] == "foton"
    assert payload["schema_version"] == "tenant_config_v1"
    assert payload["constant_to_update"] == "EXPECTED_TENANT_CONFIG_SHA256"
