from __future__ import annotations

import json
from pathlib import Path

import pytest

from mango_mvp.productization.tenant_config import load_tenant_config, validate_tenant_config


def _valid_config() -> dict:
    return {
        "schema_version": "tenant_config_v1",
        "tenant_id": "demo",
        "business": {"industry": "edtech"},
        "crm": {"target_fields": ["Авто история общения"], "protected_fields": ["Id Tallanto"]},
        "privacy": {"phone_in_ai_text": "redact"},
        "quality": {"crm_detector_min_severity": "P2"},
    }


def test_tenant_config_schema_v1_loads_and_fingerprints(tmp_path: Path) -> None:
    path = tmp_path / "tenant_config_v1.json"
    path.write_text(json.dumps(_valid_config(), ensure_ascii=False), encoding="utf-8")

    result = load_tenant_config(path)

    assert result is not None
    assert result.config["tenant_id"] == "demo"
    assert len(result.sha256) == 64


def test_tenant_config_rejects_secret_like_keys() -> None:
    payload = _valid_config()
    payload["crm"]["oauth_token"] = "secret"

    with pytest.raises(ValueError, match="must not contain secrets"):
        validate_tenant_config(payload)
