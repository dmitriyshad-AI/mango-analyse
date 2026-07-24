from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "product_data/knowledge_base/kb_release_20260612_v6_7_staging_r4_1_sources"
RELEASE_ROOT = ROOT / "product_data/knowledge_base/kb_release_20260612_v6_7_staging_r4_1"
BOT_PACK = ROOT / "product_data/knowledge_base/kb_release_20260612_v6_7_staging_r4_1_bot_pack"


def _facts(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _assert_podlipki_release(facts: list[dict[str, object]]) -> None:
    by_key = {(str(fact.get("brand") or ""), str(fact.get("fact_key") or "")): fact for fact in facts}
    podlipki = [fact for fact in facts if str(fact.get("fact_key") or "").startswith("lvsh_podlipki_2026.")]
    client = [fact for fact in podlipki if fact.get("allowed_for_client_answer") is True]
    internal = [fact for fact in podlipki if fact.get("internal_only") is True]

    assert len(client) == 14
    assert {str(fact.get("brand")) for fact in podlipki} == {"unpk"}
    assert {str(fact.get("venue")) for fact in podlipki} == {"lvsh_podlipki"}
    assert {str(fact.get("program_kind")) for fact in podlipki} == {"camp_lvsh"}

    client_blob = "\n".join(str(fact.get("client_safe_text") or "") for fact in client)
    assert "130 000 ₽" in client_blob
    assert "114 000 ₽" not in client_blob
    assert all(term not in client_blob for term in ("Фотон", "ЦДПО", "ЦРДО", "Т-Банк", "Долями"))
    assert "Группа 10 класса откроется после подтверждения достаточного набора" in client_blob

    assert {str(fact.get("fact_key")) for fact in internal} == {
        "lvsh_podlipki_2026.discounts.internal",
        "lvsh_podlipki_2026.do_not_promise.internal",
        "lvsh_podlipki_2026.pricing.minimum_internal",
    }
    assert all(not str(fact.get("client_safe_text") or "") for fact in internal)

    enrollment = by_key[("unpk", "processes_2026_06_10.unpk.camp_enrollment")]
    assert "Менделеево распродана" in str(enrollment.get("client_safe_text") or "")
    assert "Подлипки" in str(enrollment.get("client_safe_text") or "")
    boarding = by_key[("unpk", "r4_1_owner_2026_06_12.unpk.no_boarding_except_lvsh")]
    assert "Менделеево" in str(boarding.get("client_safe_text") or "")
    assert "Подлипки" in str(boarding.get("client_safe_text") or "")

    availability = by_key[("unpk", "lvsh_podlipki_2026.availability")]
    assert availability.get("valid_until") == "2026-08-15"
    assert availability.get("freshness_check_date") == "2026-07-16"


def test_pinned_release_and_bot_pack_contain_safe_podlipki_facts() -> None:
    release_facts = _facts(RELEASE_ROOT / "facts_registry.jsonl")
    bot_facts = _facts(BOT_PACK / "client_safe_facts_unpk.jsonl")
    _assert_podlipki_release(release_facts)
    assert len([fact for fact in bot_facts if str(fact.get("fact_key") or "").startswith("lvsh_podlipki_2026.")]) == 14
    assert not any(str(fact.get("fact_key") or "").endswith(".internal") for fact in bot_facts)
    gold = json.loads((BOT_PACK / "bot_gold_answers.json").read_text(encoding="utf-8"))
    camp_gold = gold["topics"]["camps"]["unpk"]
    assert "130 000 ₽" in camp_gold["gold_answer_example"]
    assert "Подлипки" in camp_gold["gold_answer_example"]
    assert "114 000 ₽" not in camp_gold["gold_answer_example"]


def test_rebuild_preserves_scope_axes_and_owner_source(tmp_path: Path) -> None:
    release = tmp_path / "release"
    handoff = tmp_path / "handoff"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/build_kb_release_v6_1_team_answers.py"),
            "--source-root", str(SOURCE_ROOT),
            "--source-out", str(SOURCE_ROOT),
            "--run-id", "podlipki_rebuild_test",
            "--release-out", str(release),
            "--handoff-out", str(handoff),
            "--bot-pack-out", str(tmp_path / "bot"),
            "--employee-pack-out", str(tmp_path / "employee"),
            "--smoke-dir", str(tmp_path / "smoke"),
        ],
        check=True,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    facts = _facts(release / "facts_registry.jsonl")
    _assert_podlipki_release(facts)
    by_key = {(str(fact.get("brand") or ""), str(fact.get("fact_key") or "")): fact for fact in facts}
    assert by_key[("unpk", "lvsh_mendeleevo_2026.location.name")]["venue"] == "lvsh_mendeleevo"
    sources = json.loads((release / "source_registry.json").read_text(encoding="utf-8"))["items"]
    source = next(item for item in sources if item.get("source_id") == "owner_google_doc:podlipki_2026")
    assert source["url"].endswith("tab=t.gnwj6o2a1spz")
