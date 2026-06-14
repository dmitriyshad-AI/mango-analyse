from __future__ import annotations

import json
from pathlib import Path

from mango_mvp.graphify_structural import (
    GRAPHIFY_COMMIT,
    classify_structured_scope,
    curated_guidance,
    curated_source_hints,
    current_revision,
    graphify_env,
    graph_source_hints,
    prepare_structural_source,
    stale_banner,
)


def test_prepare_structural_source_is_code_only_and_marks_kb_scopes(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    revision = current_revision(repo_root)

    prepared = prepare_structural_source(
        repo_root=repo_root,
        revision=revision,
        clean_archive_dir=tmp_path / "archive",
        source_dir=tmp_path / "source",
    )

    assert "src/mango_mvp/channels/rules_engine.py" in prepared.copied_code_files
    assert "src/mango_mvp/channels/rules_registry.yaml" not in prepared.copied_code_files
    non_code = [path for path in prepared.source_dir.rglob("*") if path.is_file() and path.suffix != ".py"]
    assert non_code == []
    assert any(item["scope"] == "client_safe_candidate" for item in prepared.indexed_structured_files)
    assert any(item["scope"] == "manager_only_or_internal" for item in prepared.indexed_structured_files)
    index_text = (prepared.source_dir / "src/mango_mvp/graphify_structural_index.py").read_text(encoding="utf-8")
    assert "graphify_structural_scope_rules" in index_text
    assert "semantic_cloud_layer" in index_text


def test_internal_markers_are_never_classified_client_safe() -> None:
    assert classify_structured_scope("x/CLIENT_SAFE_FACTS_FOTON.csv", "safe text") == "client_safe_candidate"
    assert (
        classify_structured_scope("x/CLIENT_SAFE_FACTS_FOTON.csv", "internal_only manager_only_route")
        == "manager_only_or_internal"
    )
    assert classify_structured_scope("x/MANAGER_ONLY_FACTS.csv", "text") == "manager_only_or_internal"


def test_graphify_environment_removes_cloud_keys(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak")
    monkeypatch.setenv("GRAPHIFY_OPENAI_MODEL", "must-not-run")
    env = graphify_env()
    assert "OPENAI_API_KEY" not in env
    assert "GRAPHIFY_OPENAI_MODEL" not in env
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"


def test_stale_banner_for_revision_mismatch(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    graph_dir = tmp_path / "graphify-out"
    graph_dir.mkdir()
    graph_path = graph_dir / "graph.json"
    graph_path.write_text(json.dumps({"nodes": [], "edges": []}), encoding="utf-8")
    (graph_dir / "mango_structural_manifest.json").write_text(
        json.dumps({"revision": "0" * 40}, sort_keys=True),
        encoding="utf-8",
    )
    assert "ВНИМАНИЕ" in stale_banner(repo_root, graph_path)


def test_pinned_graphify_commit_constant() -> None:
    assert GRAPHIFY_COMMIT == "fd470faeee16e9f42e3f47204824a2002a1f899c"


def test_curated_hints_return_raw_sources_for_p0_brand_and_script_questions() -> None:
    p0_hints = curated_source_hints("Где блокируется P0 по возврату и спорной оплате?")
    assert "src/mango_mvp/channels/rules_engine.py" in p0_hints
    assert "src/mango_mvp/channels/p0_recall_spec.py" in p0_hints

    brand_hints = curated_source_hints("Где разделяются бренды Фотон/УНПК и цены?")
    assert "src/mango_mvp/channels/rules_engine.py" in brand_hints
    assert any(path.endswith("brand_rules.yaml") for path in brand_hints)
    assert any(path.endswith("MANAGER_ONLY_FACTS.csv") for path in brand_hints)

    script_hints = curated_source_hints("Где описаны безопасные и опасные скрипты?")
    assert "docs/SCRIPT_SAFETY_MATRIX.md" in script_hints
    assert any("Graphify" in note or "read-only" in note for note in curated_guidance("Запусти Graphify HTTP/write/update"))
    assert any(
        "Cloud semantic layer не запускался" in note
        for note in curated_guidance("Перескажи threat model/gold answers из облачного слоя.")
    )


def test_graph_source_hints_normalize_archive_paths_and_append_curated_hints(tmp_path: Path) -> None:
    graph_dir = tmp_path / "graphify-out"
    source_dir = tmp_path / "source_code_only"
    graph_dir.mkdir()
    source_dir.mkdir()
    graph_path = graph_dir / "graph.json"
    graph_path.write_text(
        json.dumps(
            {
                "nodes": [
                    {
                        "id": "1",
                        "label": "rules_engine.py",
                        "source_file": str(source_dir / "src/mango_mvp/channels/rules_engine.py"),
                    }
                ],
                "edges": [],
            }
        ),
        encoding="utf-8",
    )
    (graph_dir / "mango_structural_manifest.json").write_text(
        json.dumps({"revision": current_revision(Path(__file__).resolve().parents[1]), "source_dir": str(source_dir)}),
        encoding="utf-8",
    )

    hints = graph_source_hints(graph_path, "rules_engine P0 возврат", limit=8)
    assert "src/mango_mvp/channels/rules_engine.py" in hints
    assert all("source_code_only" not in hint for hint in hints)
