from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.extract_kc_google_doc_facts import extract_google_doc_facts


def test_extract_google_doc_facts_keeps_all_candidates_unverified(tmp_path: Path) -> None:
    input_dir = tmp_path / "source_exports"
    out_dir = tmp_path / "out"
    input_dir.mkdir()
    (input_dir / "unpk_prices_2026_2027.txt").write_text(
        "\n".join(
            [
                "Стоимость обучения и порядок оплаты на 2026/2027 уч.г.",
                "5-11 класс",
                "49 000 руб.",
                "Акции подробно",
                "20% на второй и последующие предметы",
                "Договор заключается по оферте",
                "Расписание: 09:45 - 11:45 утренний клуб",
            ]
        ),
        encoding="utf-8",
    )

    result = extract_google_doc_facts(input_dir=input_dir, out_dir=out_dir)

    assert result["candidates_total"] >= 4
    rows = [json.loads(line) for line in Path(result["jsonl_path"]).read_text(encoding="utf-8").splitlines()]
    assert {row["fact_type"] for row in rows} >= {"price", "discount", "documents", "schedule"}
    assert all(row["usable_for_precise_answer"] is False for row in rows)
    assert all(row["requires_manager_confirmation"] is True for row in rows)
    assert all(row["forbidden_for_client"] is True for row in rows)

    with Path(result["csv_path"]).open("r", encoding="utf-8", newline="") as fh:
        csv_rows = list(csv.DictReader(fh))
    assert len(csv_rows) == len(rows)
    summary = json.loads(Path(result["summary_json_path"]).read_text(encoding="utf-8"))
    assert summary["precise_answer_unlocked"] == 0
