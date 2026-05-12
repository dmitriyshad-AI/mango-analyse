from __future__ import annotations

import json
from pathlib import Path

import pytest

from mango_mvp.productization.amo_snapshot_exporter import export_amo_snapshot
from scripts import mango_office_amo_snapshot_export


class FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload)

    def json(self) -> dict:
        return self.payload


class FakeSession:
    def __init__(self) -> None:
        self.urls: list[str] = []

    def get(self, url: str, *, headers: dict, params: dict | None, timeout: int) -> FakeResponse:
        self.urls.append(url)
        assert headers["Authorization"] == "Bearer token"
        if "/contacts" in url:
            return FakeResponse(
                {
                    "_embedded": {
                        "contacts": [
                            {
                                "id": 101,
                                "name": "Contact A",
                                "responsible_user_id": 9001,
                                "custom_fields_values": [
                                    {
                                        "field_code": "PHONE",
                                        "values": [{"value": "+7 999 000 00 00"}],
                                    }
                                ],
                                "_embedded": {"leads": [{"id": 501, "name": "Lead A"}]},
                            }
                        ]
                    }
                }
            )
        return FakeResponse({"_embedded": {"leads": [{"id": 777, "name": "Lead without phones"}]}})


def test_amo_snapshot_exporter_writes_entities_without_crm_write(tmp_path: Path) -> None:
    product_root = tmp_path / "product_appliance"
    product_root.mkdir()
    out = product_root / "crm_snapshots" / "amocrm_entities.json"

    report = export_amo_snapshot(
        product_root=product_root,
        output_path=out,
        base_url="https://example.amocrm.ru",
        access_token="token",
        session=FakeSession(),
    )

    assert report["summary"]["validation_ok"] is True
    assert report["summary"]["contacts_seen"] == 1
    assert report["summary"]["entities_exported"] == 1
    assert report["entities"][0]["entity_type"] == "lead"
    assert report["entities"][0]["phones"] == ["+79990000000"]
    assert report["safety"]["live_crm_reads"] is True
    assert report["safety"]["write_crm"] is False
    assert out.exists()


def test_amo_snapshot_exporter_cli_requires_credentials(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    product_root = tmp_path / "product_appliance"
    product_root.mkdir()
    monkeypatch.delenv("CRM_AMO_BASE_URL", raising=False)
    monkeypatch.delenv("AMOCRM_BASE_URL", raising=False)
    monkeypatch.delenv("CRM_AMO_API_TOKEN", raising=False)
    monkeypatch.delenv("AMOCRM_ACCESS_TOKEN", raising=False)

    rc = mango_office_amo_snapshot_export.main(["--product-root", str(product_root)])

    assert rc == 2


def test_amo_snapshot_exporter_refuses_output_outside_product_root(tmp_path: Path) -> None:
    product_root = tmp_path / "product_appliance"
    product_root.mkdir()

    with pytest.raises(ValueError, match="AMO snapshot output"):
        export_amo_snapshot(
            product_root=product_root,
            output_path=tmp_path / "outside.json",
            base_url="https://example.amocrm.ru",
            access_token="token",
            session=FakeSession(),
        )
