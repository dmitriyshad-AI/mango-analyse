from __future__ import annotations

import json
from pathlib import Path

from scripts import summarize_faithfulness_shadow as shadow_summary


def test_summarize_faithfulness_shadow_reads_direct_and_nested_events(tmp_path: Path) -> None:
    path = tmp_path / "dynamic_dialog_transcripts.jsonl"
    rows = [
        {
            "dialog_id": "direct",
            "turns": [
                {
                    "turn": 1,
                    "bot_faithfulness_shadow": [
                        {
                            "site": "main_draft",
                            "available": False,
                            "unsupported": [],
                            "verdicts": [],
                        }
                    ],
                }
            ],
        },
        {
            "dialog_id": "nested",
            "turns": [
                {
                    "turn": 2,
                    "bot_dialogue_contract_pipeline": {
                        "faithfulness_shadow": [
                            {
                                "site": "partial_yield",
                                "available": True,
                                "unsupported": ["claim"],
                                "verdicts": [{"claim": "claim", "verdict": "unsupported"}],
                            }
                        ]
                    },
                }
            ],
        },
    ]
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")

    summary = shadow_summary.summarize_file(path)
    rendered = shadow_summary.render_summary(summary)

    assert summary["by_site"] == {"main_draft": 1, "partial_yield": 1}
    assert summary["unavailable_by_site"] == {"main_draft": 1}
    assert summary["by_verdict"]["main_draft:no_claims"] == 1
    assert summary["by_verdict"]["partial_yield:unsupported"] == 1
    assert "main_draft\tno_claims\tfalse\tdirect\t1" in rendered
    assert "partial_yield\tunsupported\ttrue\tnested\t2" in rendered
