from __future__ import annotations

import importlib
import inspect
import json
from pathlib import Path
from typing import Any, Mapping

import pytest


API_MODULE = "scripts.build_kb_release_v2_from_claude_and_codex"


@pytest.fixture
def api_module() -> Any:
    try:
        return importlib.import_module(API_MODULE)
    except ModuleNotFoundError as exc:
        pytest.fail(f"Expected import API module {API_MODULE!r} to exist: {exc}")


@pytest.fixture
def kb_v2_inputs(tmp_path: Path) -> Mapping[str, Path]:
    claude_yaml_path = tmp_path / "facts_for_bot.yaml"
    claude_yaml_path.write_text(
        """
schema_version: "kb_facts_v3_FINAL_2026_05_17"
generated_at: "2026-05-17"

critical_rules:
  brand_separation:
    rule: "Для клиента УНПК МФТИ и Фотон — разные организации, не связанные"
    source: "База знаний КЦ.docx"
    severity: critical
    status: verified

neutral_learning_policy:
  status: verified
  brand: both
  product: regular_courses
  source: "База знаний КЦ.docx"
  fact_text: "Занятия помогают школьнику подобрать уровень группы после входного тестирования."

matkap:
  status: verified
  brand: both
  product: matkap
  source: "СФР, Госуслуги, БЗ КЦ"
  brands:
    foton:
      legal_entity: "ООО «ЦДПО Фотон»"
      required_docs: ["паспорт родителя", "СНИЛС", "сертификат маткапитала"]
    unpk:
      legal_entity: "АНО ДПО «УНПК МФТИ»"
      required_docs: ["паспорт родителя", "СНИЛС", "сертификат маткапитала"]

tax_deduction:
  status: verified
  brand: both
  product: tax
  source: "ФНС, БЗ КЦ"
  brands:
    foton:
      certificate_form: "КНД 1151158"
      max_return_per_child: 14300
    unpk:
      certificate_form: "КНД 1151158"
      max_return_per_child: 14300

lvsh_mendeleevo_2026:
  status: verified
  product: lvsh_mendeleevo
  source: "kmipt.ru/courses/Kanikuly/Letnyaya_vyezdnaya_fizikomatematicheskaya_shkola_8__11_kl/"
  brands:
    unpk:
      dates: "19-29 июня"
      price: 89900
      legal_entity: "АНО ДПО «УНПК МФТИ»"
    foton:
      dates: "3-14 августа"
      price: 59000
      legal_entity: "ООО «ЦДПО Фотон»"

cross_brand_script:
  status: verified
  brand: both
  product: regular_courses
  source: "База знаний КЦ.docx"
  fact_text: "В УНПК год стоит 82 000 ₽, а в Фотоне 74 500 ₽; можно предложить клиенту другой бренд."
""".strip()
        + "\n",
        encoding="utf-8",
    )

    codex_snapshot_path = tmp_path / "kc_snapshot_kb_release_20260517_v1.json"
    codex_snapshot_path.write_text(
        json.dumps(
            {
                "schema_version": "kc_knowledge_snapshot_v1",
                "summary": {
                    "sources_total": 3,
                    "knowledge_chunks_total": 1764,
                    "questions_total": 9969,
                    "answer_templates_total": 500,
                    "manager_patterns_total": 63,
                },
                "facts": [],
                "question_catalog": [{"question_id": "q1", "text": "Можно оплатить маткапиталом?"}],
                "answer_templates": [{"template_id": "t1", "text": "Менеджер уточнит."}],
                "manager_answer_patterns": [{"pattern_id": "p1", "text": "Сначала уточнить бренд."}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    source_inventory_path = tmp_path / "source_inventory.json"
    source_inventory_path.write_text(
        json.dumps(
            [
                {
                    "source_id": "source:gdrive:kc_base",
                    "title": "База знаний КЦ.docx",
                    "path": "source_exports/local_docx/kc_knowledge_base_docx.txt",
                    "url": "https://docs.google.com/document/d/kc-base",
                    "source_sha256": "a" * 64,
                    "processing_status": "processed",
                    "source_status": "read",
                },
                {
                    "source_id": "source:official:sfr_matkap",
                    "title": "СФР, Госуслуги, БЗ КЦ",
                    "path": "source_exports/official/matkap_tax.txt",
                    "url": "https://sfr.gov.ru/",
                    "source_sha256": "b" * 64,
                    "processing_status": "processed",
                    "source_status": "read",
                },
                {
                    "source_id": "source:official:fns_tax",
                    "title": "ФНС, БЗ КЦ",
                    "path": "source_exports/official/fns_tax.txt",
                    "url": "https://www.nalog.gov.ru/",
                    "source_sha256": "c" * 64,
                    "processing_status": "processed",
                    "source_status": "read",
                },
                {
                    "source_id": "source:site:kmipt_mendeleevo",
                    "title": "ЛВШ Менделеево",
                    "path": "source_exports/site_extracts/kmipt_mendeleevo.txt",
                    "url": (
                        "https://kmipt.ru/courses/Kanikuly/"
                        "Letnyaya_vyezdnaya_fizikomatematicheskaya_shkola_8__11_kl/"
                    ),
                    "source_sha256": "d" * 64,
                    "processing_status": "processed",
                    "source_status": "read",
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return {
        "claude_yaml_path": claude_yaml_path,
        "codex_snapshot_path": codex_snapshot_path,
        "source_inventory_path": source_inventory_path,
        "out_dir": tmp_path / "kb_release_v2",
        "agent_pack_dir": tmp_path / "kb_release_v2_agent_pack",
    }


def test_import_claude_layer_preserves_top_level_sections(api_module: Any, kb_v2_inputs: Mapping[str, Path]) -> None:
    layer = _load_claude_layer(api_module, kb_v2_inputs["claude_yaml_path"])
    section_names = _section_names(layer)

    assert {
        "critical_rules",
        "matkap",
        "tax_deduction",
        "lvsh_mendeleevo_2026",
        "cross_brand_script",
    }.issubset(section_names)
    assert _get_path(layer, "raw", "critical_rules", "brand_separation") or _get_path(
        layer,
        "critical_rules",
        "brand_separation",
    )


def test_normalize_v2_fact_uses_brand_neutral_instead_of_both(api_module: Any) -> None:
    fact = _normalize_v2_fact(
        api_module,
        {
            "fact_key": "regular_courses.placement_test",
            "fact_type": "policy",
            "title": "Входное тестирование",
            "fact_text": "Входное тестирование помогает подобрать уровень группы.",
            "brand": "both",
            "product": "regular_courses",
            "source": "База знаний КЦ.docx",
        },
    )

    assert fact.get("brand") == "brand_neutral"
    assert fact.get("active_brand_scope") == "brand_neutral"
    assert "both" not in {str(fact.get("brand")), str(fact.get("active_brand_scope"))}
    assert fact.get("cross_brand_policy") == "brand_neutral_allowed"


def test_matkap_and_tax_are_split_for_both_brands(api_module: Any, kb_v2_inputs: Mapping[str, Path]) -> None:
    facts = _build_facts(api_module, kb_v2_inputs)

    matkap_facts = _facts_matching(facts, "matkap", "маткап")
    tax_facts = _facts_matching(facts, "tax", "налог", "вычет")

    assert _brands(matkap_facts) == {"foton", "unpk"}
    assert _brands(tax_facts) == {"foton", "unpk"}
    assert all(fact.get("brand") != "brand_neutral" for fact in matkap_facts + tax_facts)
    assert all(fact.get("allowed_for_client_answer") is not True for fact in matkap_facts + tax_facts)


def test_mendeleevo_lvsh_is_split_by_brand(api_module: Any, kb_v2_inputs: Mapping[str, Path]) -> None:
    facts = _build_facts(api_module, kb_v2_inputs)
    mendeleevo_facts = _facts_matching(facts, "mendeleevo", "менделеево", "lvsh")

    assert _brands(mendeleevo_facts) == {"foton", "unpk"}
    for brand in ("foton", "unpk"):
        same_brand = [fact for fact in mendeleevo_facts if fact.get("brand") == brand]
        other_brand = "unpk" if brand == "foton" else "foton"
        assert same_brand
        assert all(fact.get("active_brand_scope") in {f"{brand}_bot", brand} for fact in same_brand)
        assert all(fact.get("brand") != other_brand for fact in same_brand)


def test_cross_brand_mixed_facts_are_internal_only(api_module: Any) -> None:
    fact = _normalize_v2_fact(
        api_module,
        {
            "fact_key": "regular_courses.cross_brand_comparison",
            "fact_type": "policy",
            "title": "Смешанное сравнение брендов",
            "fact_text": "В УНПК год стоит 82 000 ₽, а в Фотоне 74 500 ₽.",
            "brand": "both",
            "product": "regular_courses",
            "source": "База знаний КЦ.docx",
        },
    )

    assert fact.get("active_brand_scope") == "internal_only"
    assert fact.get("allowed_for_client_answer") is False
    assert fact.get("cross_brand_policy") == "forbidden_for_client"
    assert fact.get("cross_brand_mixed") is True
    assert fact.get("brand") == "internal"


def test_imported_facts_keep_source_linkage(api_module: Any, kb_v2_inputs: Mapping[str, Path]) -> None:
    facts = _build_facts(api_module, kb_v2_inputs)
    checked_facts = _facts_matching(facts, "matkap", "tax", "налог", "mendeleevo", "менделеево")

    assert checked_facts
    for fact in checked_facts:
        assert fact.get("source_id"), fact
        assert fact.get("source_title"), fact
        assert fact.get("source_path"), fact
        assert fact.get("source_sha256"), fact
        assert fact.get("source_status"), fact


def _load_claude_layer(api_module: Any, path: Path) -> Any:
    func = _required_func(api_module, "load_claude_layer")
    return _call_with_supported_kwargs(
        func,
        {
            "claude_yaml_path": path,
            "claude_facts_path": path,
            "facts_yaml_path": path,
            "yaml_path": path,
            "path": path,
        },
    )


def _normalize_v2_fact(api_module: Any, raw_fact: Mapping[str, Any]) -> Mapping[str, Any]:
    func = _required_func(api_module, "normalize_v2_fact")
    source_record = {
        "source_id": "source:gdrive:kc_base",
        "source_title": "База знаний КЦ.docx",
        "source_path": "source_exports/local_docx/kc_knowledge_base_docx.txt",
        "source_sha256": "a" * 64,
        "source_status": "read",
    }
    result = _call_with_supported_kwargs(
        func,
        {
            "raw_fact": raw_fact,
            "fact": raw_fact,
            "record": raw_fact,
            "section_name": raw_fact.get("fact_key", "test.fact"),
            "fact_key": raw_fact.get("fact_key", "test.fact"),
            "raw_value": raw_fact,
            "source_lookup": {"База знаний КЦ.docx": source_record},
            "source_record": source_record,
        },
    )
    assert isinstance(result, Mapping)
    return result


def _build_facts(api_module: Any, paths: Mapping[str, Path]) -> list[Mapping[str, Any]]:
    func = _required_func(api_module, "build_kb_release_v2")
    result = _call_with_supported_kwargs(
        func,
        {
            "claude_yaml_path": paths["claude_yaml_path"],
            "claude_facts_path": paths["claude_yaml_path"],
            "facts_yaml_path": paths["claude_yaml_path"],
            "codex_snapshot_path": paths["codex_snapshot_path"],
            "snapshot_path": paths["codex_snapshot_path"],
            "v1_snapshot_path": paths["codex_snapshot_path"],
            "codex_source_inventory_path": paths["source_inventory_path"],
            "source_inventory_path": paths["source_inventory_path"],
            "inventory_path": paths["source_inventory_path"],
            "out_dir": paths["out_dir"],
            "output_dir": paths["out_dir"],
            "release_dir": paths["out_dir"],
            "agent_pack_dir": paths["agent_pack_dir"],
            "run_id": "test_kb_release_v2",
        },
    )
    snapshot = _snapshot_from_build_result(result, paths["out_dir"])
    facts = _extract_facts(snapshot)
    assert facts
    return facts


def _required_func(api_module: Any, name: str) -> Any:
    func = getattr(api_module, name, None)
    if func is None:
        pytest.fail(f"Expected {API_MODULE}.{name} to exist")
    return func


def _call_with_supported_kwargs(func: Any, values: Mapping[str, Any]) -> Any:
    signature = inspect.signature(func)
    params = signature.parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
        return func(**values)

    kwargs = {
        name: values[name]
        for name in params
        if name in values and params[name].kind in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
    }
    missing_required = [
        name
        for name, param in params.items()
        if param.default is inspect.Parameter.empty
        and param.kind in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
        and name not in kwargs
    ]
    if missing_required:
        if len(params) == 1 and "path" in values:
            return func(values["path"])
        pytest.fail(f"Unsupported signature for {func.__name__}: missing values for {missing_required}")
    return func(**kwargs)


def _snapshot_from_build_result(result: Any, out_dir: Path) -> Any:
    if isinstance(result, Mapping):
        if "facts" in result or "facts_registry" in result:
            return result
        if isinstance(result.get("snapshot"), Mapping):
            return result["snapshot"]
        for key in ("snapshot_path", "kb_release_v2_snapshot_path", "facts_registry_json_path"):
            if result.get(key):
                path = Path(result[key])
                if path.exists():
                    return json.loads(path.read_text(encoding="utf-8"))
        if result.get("out_dir"):
            out_dir = Path(result["out_dir"])
    elif isinstance(result, (str, Path)):
        path = Path(result)
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
        if path.is_dir():
            out_dir = path

    for candidate in (
        out_dir / "kb_release_v2_snapshot.json",
        out_dir / "facts_registry.json",
    ):
        if candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8"))
    jsonl_candidate = out_dir / "facts_registry.jsonl"
    if jsonl_candidate.exists():
        return [json.loads(line) for line in jsonl_candidate.read_text(encoding="utf-8").splitlines() if line.strip()]
    pytest.fail("build_kb_release_v2 did not return or write a readable v2 snapshot/facts registry")


def _extract_facts(snapshot: Any) -> list[Mapping[str, Any]]:
    if isinstance(snapshot, list):
        return [fact for fact in snapshot if isinstance(fact, Mapping)]
    if not isinstance(snapshot, Mapping):
        pytest.fail(f"Unsupported snapshot type: {type(snapshot)!r}")
    for key in ("facts_registry", "facts", "fact_records"):
        value = snapshot.get(key)
        if isinstance(value, list):
            return [fact for fact in value if isinstance(fact, Mapping)]
    if isinstance(snapshot.get("snapshot"), Mapping):
        return _extract_facts(snapshot["snapshot"])
    pytest.fail("No facts found in kb_release_v2 build result")


def _section_names(layer: Any) -> set[str]:
    if isinstance(layer, Mapping):
        if isinstance(layer.get("top_level_sections"), list):
            return set(layer["top_level_sections"])
        if isinstance(layer.get("sections"), Mapping):
            return set(layer["sections"])
        if isinstance(layer.get("raw"), Mapping):
            return set(layer["raw"])
        return set(layer)
    pytest.fail(f"Unsupported Claude layer type: {type(layer)!r}")


def _get_path(value: Any, *path: str) -> Any:
    current = value
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _facts_matching(facts: list[Mapping[str, Any]], *needles: str) -> list[Mapping[str, Any]]:
    lowered_needles = tuple(needle.lower() for needle in needles)
    return [fact for fact in facts if any(needle in _fact_blob(fact) for needle in lowered_needles)]


def _fact_blob(fact: Mapping[str, Any]) -> str:
    fields = (
        "fact_id",
        "fact_key",
        "fact_type",
        "title",
        "fact_text",
        "client_safe_text",
        "manager_check_text",
        "product",
    )
    return " ".join(str(fact.get(field, "")) for field in fields).lower()


def _brands(facts: list[Mapping[str, Any]]) -> set[str]:
    return {str(fact.get("brand")) for fact in facts if fact.get("brand") in {"foton", "unpk"}}
