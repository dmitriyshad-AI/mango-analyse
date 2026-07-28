from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "scan_single_owner.py"
CONCEPT_LIMITS = {
    "normalize_active_brand": {
        "names": {"normalize_active_brand", "_normalize_active_brand", "_normalize_brand"},
        "limit": 4,
    },
    "optional_text": {"names": {"optional_text", "_optional_text"}, "limit": 3},
    "require_timezone": {"names": {"require_timezone", "_require_timezone"}, "limit": 2},
}


def _scan(root: Path) -> list[dict[str, object]]:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(root), "--json"],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(completed.stdout)


def _registry_violations(rows: list[dict[str, object]]) -> list[str]:
    violations: list[str] = []
    for concept, rule in CONCEPT_LIMITS.items():
        matches = [
            row for row in rows
            if row["name"] in rule["names"]
            and (
                str(row["file"]).startswith("src/mango_mvp/channels/")
                or (concept == "normalize_active_brand" and row["file"] == "src/mango_mvp/knowledge_base/kc_context.py")
            )
        ]
        if len(matches) > int(rule["limit"]):
            sites = ", ".join(f'{row["file"]}:{row["line"]}' for row in matches)
            violations.append(f"{concept}: {len(matches)} > {rule['limit']}: {sites}")
    return violations


def test_scan_classifies_all_reference_boundaries_and_reports_every_duplicate(tmp_path: Path) -> None:
    source = tmp_path / "src" / "mango_mvp"
    tests = tmp_path / "tests"
    source.mkdir(parents=True)
    tests.mkdir()
    (source / "sample.py").write_text(
        "__all__ = ['dynamic_only']\n"
        "__all__.extend(['extended_dynamic'])\n"
        "def dynamic_only(value): return str(value).strip()\n"
        "def extended_dynamic(item): return str(item).strip()\n"
        "def direct(value): return value + 1\n"
        "def copy_a(value):\n    clean = str(value).strip()\n    return clean\n"
        "def copy_b(item):\n    text = str(item).strip()\n    return text\n"
        "def _test_only(value): return value\n"
        "def _unused(value): return value\n"
        "def PublicApi(value): return value\n"
        "direct(1)\n"
        "hasattr(object(), 'extended_dynamic')\n",
        encoding="utf-8",
    )
    (tests / "test_sample.py").write_text("from mango_mvp.sample import _test_only\n_test_only(1)\n", encoding="utf-8")

    rows = _scan(tmp_path)
    by_name = {str(row["name"]): row for row in rows}

    assert by_name["dynamic_only"]["status"] == "dynamic_referenced"
    assert by_name["extended_dynamic"]["status"] == "dynamic_referenced"
    assert by_name["direct"]["status"] == "referenced"
    assert by_name["_test_only"]["status"] == "referenced_only_by_tests"
    assert by_name["_unused"]["status"] == "unreferenced"
    assert by_name["PublicApi"]["status"] == "dynamic_or_external_unknown"
    assert by_name["copy_a"]["ast_hash"] == by_name["copy_b"]["ast_hash"]
    assert by_name["copy_a"]["duplicate_sites"] == [
        "src/mango_mvp/sample.py:6",
        "src/mango_mvp/sample.py:9",
    ]

def test_live_inventory_keeps_dynamic_reexport_alive() -> None:
    repo = Path(__file__).resolve().parents[1]
    rows = _scan(repo)
    provider = next(
        row for row in rows
        if row["file"] == "src/mango_mvp/channels/subscription_llm_parts/provider.py"
        and row["name"] == "_optional_text"
    )

    assert provider["status"] == "dynamic_referenced"
    assert any("subscription_llm_parts/__init__.py" in str(site) for site in provider["dynamic_sites"])
    assert _registry_violations(rows) == []
    assert len(CONCEPT_LIMITS) <= 3, "A new concept needs its own dedup commit, not a higher baseline"

    injected = dict(provider, file="src/mango_mvp/channels/injected_duplicate.py", line=1)
    violations = _registry_violations([*rows, injected])
    assert violations == [
        "optional_text: 4 > 3: "
        "src/mango_mvp/channels/contracts.py:391, "
        "src/mango_mvp/channels/subscription_llm_parts/provider.py:2731, "
        "src/mango_mvp/channels/telegram_pilot_store.py:919, "
        "src/mango_mvp/channels/injected_duplicate.py:1"
    ]
