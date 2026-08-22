from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from scripts.build_kb_partner_pack import build_partner_pack


def test_partner_pack_exports_only_whitelisted_client_safe_fields(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(json.dumps({"run_id": "kb-v6.8", "facts": [
        _fact("f1", "foton", "bot_answer_self_for_pilot"),
        _fact("u1", "unpk", "draft_for_manager"),
        {**_fact("handoff", "foton", "bot_answer_self_for_pilot"), "client_safe_text": "Я уже передал запрос менеджеру."},
        {**_fact("owner-marker", "foton", "draft_for_manager"), "client_safe_text": "Подтверждено Дмитрием 2026-05-20."},
        {**_fact("secret", "foton", "manager_only"), "allowed_for_client_answer": False, "internal_text": "секрет"},
    ]}, ensure_ascii=False), encoding="utf-8")
    out = tmp_path / "partner"

    manifest = build_partner_pack(snapshot, out)

    foton = _jsonl(out / "kb_foton.jsonl")
    unpk = _jsonl(out / "kb_unpk.jsonl")
    assert [item["fact_id"] for item in foton] == ["f1", "handoff"]
    assert [item["fact_id"] for item in unpk] == ["u1"]
    assert foton[0]["usage_mode"] == "answer_source"
    assert unpk[0]["usage_mode"] == "manager_review_required"
    assert foton[1]["client_safe_text"].startswith("По возврату после оплаты нужен расчёт менеджера")
    assert foton[1]["usage_mode"] == "manager_review_required"
    assert isinstance(foton[0]["bot_template_required"], bool)
    assert manifest["excluded_by_external_policy"] == 1
    assert manifest["completed_action_claims_neutralized"] == 1
    assert manifest["excluded_non_client_safe_facts"] == 1
    assert manifest["source_client_safe_facts"] == 4
    forbidden = {"fact_text", "internal_text", "manager_check_text", "manager_display_text", "source_path", "source_id", "notes", "route_policy"}
    assert not forbidden.intersection(foton[0])
    assert "секрет" not in "".join(path.read_text(encoding="utf-8", errors="ignore") for path in out.iterdir())
    assert "manager_review_required" not in (out / "KB_FOTON_AUTO.md").read_text(encoding="utf-8")
    schema = json.loads((out / "schema.json").read_text(encoding="utf-8"))
    assert schema["additionalProperties"] is False
    assert schema["properties"]["usage_mode"]["enum"] == ["answer_source", "manager_review_required"]
    _assert_checksums(out)


def test_partner_pack_rejects_broken_client_safe_permission(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(json.dumps({"facts": [{**_fact("bad", "foton", "bot_answer_self_for_pilot"), "internal_only": True}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="permission contract"):
        build_partner_pack(snapshot, tmp_path / "out")


def test_partner_pack_current_v6_8_counts_and_brand_isolation(tmp_path: Path) -> None:
    source = Path("product_data/knowledge_base/kb_release_20260813_v6_8_owner_approved/kb_release_v3_snapshot.json")
    out = tmp_path / "current"
    manifest = build_partner_pack(source, out)
    assert manifest["facts_total"] == 470
    assert manifest["facts_by_brand"] == {"foton": 197, "unpk": 273}
    assert manifest["usage_modes"] == {"answer_source": 310, "manager_review_required": 160}
    assert manifest["completed_action_claims_neutralized"] == 2
    assert manifest["excluded_non_client_safe_facts"] == 335
    assert manifest["excluded_by_external_policy"] == 1
    records = _jsonl(out / "kb_foton.jsonl") + _jsonl(out / "kb_unpk.jsonl")
    assert sum(item["refresh_before_launch"] is True for item in records) == 239
    assert not any(item["usage_mode"] == "answer_source" and any(marker in str(item["client_safe_text"]).casefold() for marker in ("возврат", "возвращается", "вернуть за него", "средства не сгорают")) for item in records)
    assert not any("я уже передал" in str(item["client_safe_text"]).casefold() for item in records)
    assert not any("дмитри" in str(item["client_safe_text"]).casefold() for item in records)
    assert all(item["brand"] == "foton" for item in _jsonl(out / "kb_foton.jsonl"))
    assert all(item["brand"] == "unpk" for item in _jsonl(out / "kb_unpk.jsonl"))


def test_partner_guide_matches_manifest_and_has_no_internal_markers() -> None:
    root = Path("product_data/knowledge_base/kb_release_20260813_v6_8_partner_pack")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    with zipfile.ZipFile(root / "README_FOR_PARTNERS.docx") as archive:
        xml = "\n".join(archive.read(name).decode("utf-8", "ignore") for name in archive.namelist() if name.endswith(".xml"))
    assert str(manifest["facts_total"]) in xml
    assert str(manifest["facts_by_brand"]["foton"]) in xml
    assert str(manifest["facts_by_brand"]["unpk"]) in xml
    assert not any(marker in xml.casefold() for marker in ("/users/", ".mango_local", "подтверждено дмитрием", "я уже передал", "route_policy"))


def _fact(fact_id: str, brand: str, route: str) -> dict[str, object]:
    return {
        "fact_id": fact_id, "brand": brand, "fact_type": "price", "product": "course",
        "client_safe_text": f"{brand}: подтверждённый факт.", "route_policy": route,
        "risk_level": "low", "freshness_status": "document_verified", "valid_from": "2026-08-13",
        "valid_until": "2027-05-31", "bot_template_required": False,
        "allowed_for_client_answer": True, "internal_only": False, "forbidden_for_client": False,
    }


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _assert_checksums(root: Path) -> None:
    for line in (root / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected
