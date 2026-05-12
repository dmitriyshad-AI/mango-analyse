from __future__ import annotations

import json
from pathlib import Path

import pytest

from mango_mvp.productization.demo_tenant import build_demo_tenant_product_root
from mango_mvp.productization.processing_acceptance_gates import build_processing_acceptance_gates_report
from scripts import mango_office_processing_acceptance_gates


def test_processing_acceptance_gates_block_until_processing_quality_is_ready(tmp_path: Path) -> None:
    product_root = tmp_path / "product_appliance"
    build_demo_tenant_product_root(product_root=product_root, replace_existing=True)

    report = build_processing_acceptance_gates_report(
        product_root,
        product_root / "mango_product_appliance.sqlite",
        out_path=product_root / "processing_acceptance_gates" / "report.json",
    )

    assert report["summary"]["validation_ok"] is False
    assert report["summary"]["processing_quality_external_ready"] is False
    assert any(gate["gate"] == "PROCESSING_QUALITY_EXTERNAL_READY" and not gate["passed"] for gate in report["gates"])
    assert report["safety"]["run_asr"] is False


def test_processing_acceptance_gates_can_pass_with_explicit_processing_evidence(tmp_path: Path) -> None:
    product_root = tmp_path / "product_appliance"
    build_demo_tenant_product_root(product_root=product_root, replace_existing=True)
    evidence = product_root / "processing_quality_acceptance.json"
    evidence.write_text('{"accepted": true}\n', encoding="utf-8")

    report = build_processing_acceptance_gates_report(
        product_root,
        product_root / "mango_product_appliance.sqlite",
        processing_quality_report_path=evidence,
    )

    assert report["summary"]["processing_quality_external_ready"] is True
    assert any(gate["gate"] == "PROCESSING_QUALITY_EXTERNAL_READY" and gate["passed"] for gate in report["gates"])


def test_processing_acceptance_gates_cli_writes_report(tmp_path: Path) -> None:
    product_root = tmp_path / "product_appliance"
    build_demo_tenant_product_root(product_root=product_root, replace_existing=True)
    out = product_root / "processing_acceptance_gates" / "report.json"

    rc = mango_office_processing_acceptance_gates.main(
        [
            "--product-root",
            str(product_root),
            "--product-db",
            str(product_root / "mango_product_appliance.sqlite"),
            "--out",
            str(out),
        ]
    )

    saved = json.loads(out.read_text(encoding="utf-8"))
    assert rc == 1
    assert saved["summary"]["processing_quality_external_ready"] is False
    assert saved["safety"]["write_crm"] is False


def test_processing_acceptance_gates_refuses_stable_runtime_evidence(tmp_path: Path) -> None:
    product_root = tmp_path / "product_appliance"
    build_demo_tenant_product_root(product_root=product_root, replace_existing=True)

    with pytest.raises(ValueError, match="stable_runtime"):
        build_processing_acceptance_gates_report(
            product_root,
            product_root / "mango_product_appliance.sqlite",
            processing_quality_report_path=tmp_path / "stable_runtime" / "quality.json",
        )
