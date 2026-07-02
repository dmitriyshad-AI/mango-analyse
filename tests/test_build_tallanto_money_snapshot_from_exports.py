from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.build_tallanto_money_snapshot_from_exports import build_snapshot


def test_build_tallanto_money_snapshot_dedupes_and_is_deterministic(tmp_path: Path) -> None:
    finance_path = tmp_path / "finances.json"
    abonement_path = tmp_path / "abonements.json"
    class_path = tmp_path / "classes.json"
    out1 = tmp_path / ".codex_local" / "staging" / "snapshot.json"
    out2 = tmp_path / ".codex_local" / "staging" / "snapshot_repeat.json"
    finance_path.write_text(
        json.dumps(
            [
                {"id": "pay-1", "contact_id": "contact-1", "cost": 1000},
                {"id": "pay-1", "contact_id": "contact-1", "cost": 1000},
                {"id": "pay-2", "contact_id": "contact-2", "cost": 2000},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    abonement_path.write_text(
        json.dumps(
            {
                "abonement-1": {"id": "abonement-1", "contact_id": "contact-1"},
                "abonement-1-copy": {"id": "abonement-1", "contact_id": "contact-1"},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    class_path.write_text(json.dumps([{"id": "class-1", "name": "Физика"}], ensure_ascii=False), encoding="utf-8")

    first = build_snapshot(
        allowed_root=tmp_path,
        output=out1,
        finance_inputs=(finance_path,),
        abonement_inputs=(abonement_path,),
        class_inputs=(class_path,),
    )
    second = build_snapshot(
        allowed_root=tmp_path,
        output=out2,
        finance_inputs=(finance_path,),
        abonement_inputs=(abonement_path,),
        class_inputs=(class_path,),
    )
    payload = json.loads(out1.read_text(encoding="utf-8"))

    assert first["output_sha256"] == second["output_sha256"]
    assert first["counts"] == {"most_finances": 2, "most_abonements": 1, "most_class": 1}
    assert payload["stats"]["most_finances"]["duplicate_id_rows"] == 1
    assert payload["stats"]["most_abonements"]["duplicate_id_rows"] == 1
    assert [row["id"] for row in payload["most_finances"]] == ["pay-1", "pay-2"]


def test_build_tallanto_money_snapshot_refuses_output_outside_staging(tmp_path: Path) -> None:
    finance_path = tmp_path / "finances.json"
    finance_path.write_text(json.dumps([{"id": "pay-1"}]), encoding="utf-8")

    with pytest.raises(ValueError, match=".codex_local/staging"):
        build_snapshot(
            allowed_root=tmp_path,
            output=tmp_path / "snapshot.json",
            finance_inputs=(finance_path,),
            abonement_inputs=(),
            class_inputs=(),
        )
