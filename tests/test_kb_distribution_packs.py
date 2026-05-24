from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.build_kb_distribution_packs import build_distribution_packs
from scripts.build_kb_release_v6_1_team_answers import gold_answers_v3_payload


def test_distribution_packs_split_employee_and_bot_outputs(tmp_path: Path) -> None:
    release = _write_release(tmp_path / "handoff")
    full_release = _write_full_release(tmp_path / "full_release")
    smoke = _write_smoke(tmp_path / "smoke")
    employee_out = tmp_path / "employee_pack"
    bot_out = tmp_path / "bot_pack"

    result = build_distribution_packs(
        release_dir=release,
        full_release_dir=full_release,
        smoke_dir=smoke,
        employee_out=employee_out,
        bot_out=bot_out,
    )

    assert result["facts_total"] == 3
    assert result["client_safe_facts"] == 2
    assert (employee_out / "START_HERE.md").exists()
    assert (employee_out / "FOR_AI_AGENTS.md").exists()
    assert (employee_out / "FOTON.md").exists()
    assert (employee_out / "UNPK.md").exists()
    assert (bot_out / "README_FOR_BOT.md").exists()
    assert (bot_out / "BOT_USAGE_CONTRACT.md").exists()
    assert (bot_out / "post_filter_registry.json").exists()
    assert (bot_out / "bot_template_registry.json").exists()
    assert not (bot_out / "kb_release_v3_snapshot.json").exists()
    assert not (bot_out / "approval_queue_for_rop_v3.csv").exists()

    foton_facts = _read_jsonl(bot_out / "client_safe_facts_foton.jsonl")
    unpk_facts = _read_jsonl(bot_out / "client_safe_facts_unpk.jsonl")
    manager_only = _read_jsonl(bot_out / "manager_only_or_internal_facts.jsonl")

    assert [item["brand"] for item in foton_facts] == ["foton"]
    assert [item["brand"] for item in unpk_facts] == ["unpk"]
    assert [item["brand"] for item in manager_only] == ["internal"]

    bot_manifest = json.loads((bot_out / "manifest.json").read_text(encoding="utf-8"))
    bot_contract = (bot_out / "BOT_USAGE_CONTRACT.md").read_text(encoding="utf-8")
    assert bot_manifest["formal_pass"] is True
    assert bot_manifest["semantic_pass"] is True
    assert bot_manifest["safety"]["client_auto_send"] is False
    assert bot_manifest["safety"]["crm_write"] is False
    assert "не подставляет `client_safe_text` дословно" in bot_contract
    assert "`bot_template_required=true`" in bot_contract
    assert "`pattern_descriptions`" in bot_contract
    assert "`phrases_by_active_brand[active_brand]`" in bot_contract
    template_registry = json.loads((bot_out / "bot_template_registry.json").read_text(encoding="utf-8"))
    required_fact_keys = {
        item["fact_key"]
        for item in foton_facts + unpk_facts
        if item.get("bot_template_required")
    }
    template_fact_keys = {item["fact_key"] for item in template_registry["templates"]}
    assert required_fact_keys <= template_fact_keys
    assert template_registry["fallback_route_if_missing"] == "manager_only"


def test_distribution_packs_render_gold_identity_policy_c(tmp_path: Path) -> None:
    release = _write_release(tmp_path / "handoff")
    snapshot_path = release / "kb_release_v3_snapshot.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    snapshot["bot_policy"] = {"gold_answers_v3": gold_answers_v3_payload()}
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False), encoding="utf-8")
    full_release = _write_full_release(tmp_path / "full_release")
    smoke = _write_smoke(tmp_path / "smoke")
    bot_out = tmp_path / "bot_pack"

    build_distribution_packs(
        release_dir=release,
        full_release_dir=full_release,
        smoke_dir=smoke,
        employee_out=tmp_path / "employee_pack",
        bot_out=bot_out,
    )

    bot_gold = json.loads((bot_out / "bot_gold_answers.json").read_text(encoding="utf-8"))
    rules_yaml = (bot_out / "gold_answer_rules.yaml").read_text(encoding="utf-8")
    markdown = (bot_out / "GOLD_ANSWERS_FOR_BOT.md").read_text(encoding="utf-8")

    assert "identity" in bot_gold["topics"]
    assert "цифровой помощник Фотона" in markdown
    assert "цифровой помощник УНПК МФТИ" in markdown
    assert "OpenAI" in rules_yaml
    assert "я человек" in rules_yaml


def test_distribution_packs_reject_stable_runtime_outputs(tmp_path: Path) -> None:
    release = _write_release(tmp_path / "handoff")
    full_release = _write_full_release(tmp_path / "full_release")
    smoke = _write_smoke(tmp_path / "smoke")

    with pytest.raises(ValueError, match="stable_runtime"):
        build_distribution_packs(
            release_dir=release,
            full_release_dir=full_release,
            smoke_dir=smoke,
            employee_out=tmp_path / "stable_runtime" / "employee",
            bot_out=tmp_path / "bot",
        )


def _write_release(root: Path) -> Path:
    root.mkdir(parents=True)
    facts = [
        {
            "fact_id": "fact:foton:price",
            "brand": "foton",
            "fact_type": "price",
            "fact_key": "prices.year",
            "allowed_for_client_answer": True,
            "client_safe_text": "Фотон: учебный год стоит 74 500 ₽.",
            "manager_check_text": "Фотон: учебный год стоит 74 500 ₽.",
            "source_title": "Прайс Фотона",
            "freshness_status": "document_verified",
            "route_policy": "draft_for_manager",
            "risk_level": "low",
            "valid_until": "2026-08-31",
            "bot_template_required": True,
        },
        {
            "fact_id": "fact:unpk:price",
            "brand": "unpk",
            "fact_type": "price",
            "fact_key": "prices.year",
            "allowed_for_client_answer": True,
            "client_safe_text": "УНПК МФТИ: учебный год стоит 84 500 ₽.",
            "manager_check_text": "УНПК МФТИ: учебный год стоит 84 500 ₽.",
            "source_title": "Прайс УНПК",
            "freshness_status": "document_verified",
            "route_policy": "draft_for_manager",
            "risk_level": "low",
            "valid_until": "2026-08-31",
        },
        {
            "fact_id": "fact:internal:promo",
            "brand": "internal",
            "fact_type": "promocode",
            "fact_key": "promocode.teacher",
            "allowed_for_client_answer": False,
            "client_safe_text": "",
            "manager_check_text": "Промокод только для внутренней проверки.",
            "source_title": "Внутренние правила",
            "freshness_status": "internal",
            "route_policy": "manager_only",
            "risk_level": "high",
            "safety_block_reasons": ["internal_only"],
        },
    ]
    (root / "facts_registry.jsonl").write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in facts) + "\n",
        encoding="utf-8",
    )
    (root / "kb_release_v3_snapshot.json").write_text(
        json.dumps({"run_id": "test-run", "facts": facts}, ensure_ascii=False),
        encoding="utf-8",
    )
    (root / "quality_report.json").write_text(json.dumps({"quality_passed": True}), encoding="utf-8")
    (root / "semantic_review.json").write_text(
        json.dumps({"semantic_pass": True, "blocking_findings": 0, "findings_total": 0}),
        encoding="utf-8",
    )
    (root / "source_registry.json").write_text("{}", encoding="utf-8")
    (root / "approval_queue_for_rop_v3.csv").write_text("priority,rop_question\n", encoding="utf-8")
    return root


def _write_full_release(root: Path) -> Path:
    root.mkdir(parents=True)
    for filename in ("post_filter_registry.json", "bot_policy.yaml", "brand_rules.yaml"):
        (root / filename).write_text("{}\n", encoding="utf-8")
    return root


def _write_smoke(root: Path) -> Path:
    for brand in ("FOTON", "UNPK"):
        brand_dir = root / brand
        brand_dir.mkdir(parents=True)
        (brand_dir / "stage6_eval_summary.json").write_text(
            json.dumps({"rows_total": 25, "errors": 0, "brand_separation_violation": 0}),
            encoding="utf-8",
        )
    return root


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
