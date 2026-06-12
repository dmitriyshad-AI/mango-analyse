from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from mango_mvp.existing_clients.run_roots import cli_run_out_root


NOW = datetime(2026, 6, 12, 16, 30, tzinfo=timezone.utc)


def test_cli_run_out_root_allocates_separate_folder_without_touching_first(tmp_path: Path) -> None:
    project = tmp_path / "project"
    base = Path("product_data/customer_profiles/tz14_scan")
    first = cli_run_out_root(project_root=project, out_root=base, generated_at=NOW)
    first.mkdir(parents=True)
    marker = first / "summary.json"
    marker.write_text("first", encoding="utf-8")

    second = cli_run_out_root(project_root=project, out_root=base, generated_at=NOW)
    second.mkdir(parents=True)
    (second / "summary.json").write_text("second", encoding="utf-8")

    assert first != second
    assert first.name == "run_20260612T163000Z"
    assert second.name == "run_20260612T163000Z_02"
    assert marker.read_text(encoding="utf-8") == "first"


def test_cli_run_out_root_supports_absolute_base(tmp_path: Path) -> None:
    base = tmp_path / "product_data" / "customer_profiles" / "tz14_scan"

    out = cli_run_out_root(project_root=tmp_path / "other", out_root=base, generated_at=NOW)

    assert out.parent == base
    assert out.name == "run_20260612T163000Z"
