from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.run_tz121_brand_e_followup_real import main as brand_e_followup_main


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_tz121_brand_e_followup_gate_uses_real_row_indexes_without_raw_fragments(tmp_path: Path) -> None:
    review = tmp_path / "review.csv"
    master = tmp_path / "master_contacts.csv"
    out_dir = tmp_path / "out"
    _write_csv(
        review,
        [
            {
                "row_index": "2",
                "flip": "foton->unknown",
                "verdict": "false_negative",
                "reason": "падежная форма Фотона",
            },
            {
                "row_index": "3",
                "flip": "unpk->unknown",
                "verdict": "expected_fail_closed",
                "reason": "оба бренда",
            },
            {
                "row_index": "4",
                "flip": "foton->unknown",
                "verdict": "expected_fail_closed",
                "reason": "небрендовое слово",
            },
        ],
    )
    _write_csv(
        master,
        [
            {"История": "клиент занимался у Фотона", "Филиал Tallanto": "МФТИ"},
            {"История": "Фотон и УНПК в одном вопросе", "Филиал Tallanto": ""},
            {"История": "мотивация через фотончики", "Филиал Tallanto": ""},
        ],
    )

    assert brand_e_followup_main(["--review", str(review), "--master-contacts", str(master), "--out-dir", str(out_dir)]) == 0

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["gate_passed"] is True
    assert summary["foton_false_negative_fixed"] == 1
    assert summary["expected_fail_closed_kept_unknown"] == 2
    assert summary["llm_calls_total"] == 0
    assert summary["safety"]["writes_raw_pii_to_git"] is False

    rows = list(csv.DictReader((out_dir / "tz121_e_brand_followup_trace.csv").open(encoding="utf-8-sig")))
    assert all(row["input_fragment"] == "real master_contacts row redacted; see local ignored source" for row in rows)
    assert rows[0]["model"] == "foton"
    assert rows[1]["model"] == "unknown"
    assert rows[2]["model"] == "unknown"
