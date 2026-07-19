from pathlib import Path

import pytest

from mango_mvp.customer_timeline.full_memory_ingest import FullMemoryIngestConfig
from mango_mvp.productization.mail_archive import (
    CANONICAL_MAIL_ARCHIVE_DB,
    CANONICAL_MAIL_ARCHIVE_ROOT,
    CANONICAL_MAIL_IDENTITY_DB,
    CANONICAL_MAIL_STAGE2_DELTA_EVENTS,
    CANONICAL_MAIL_STAGE2_FULL_EVENTS,
    DEFAULT_MAIL_DATA_ROOT,
)
from mango_mvp.question_catalog.builder import default_config
from scripts.email_pipeline.archive_sources import default_archive_specs
from scripts.import_mail_bridge_to_customer_timeline import (
    build_parser as build_mail_bridge_parser,
    default_corpus_events,
    default_delta_events,
    default_identity_db,
    run_bridge,
)


def test_default_mail_readers_use_one_canonical_archive(tmp_path: Path) -> None:
    assert default_archive_specs(tmp_path)[0].path == tmp_path / CANONICAL_MAIL_ARCHIVE_DB
    assert default_archive_specs()[0].path == DEFAULT_MAIL_DATA_ROOT / CANONICAL_MAIL_ARCHIVE_DB
    assert default_config(tmp_path).mail_archive_root == (
        DEFAULT_MAIL_DATA_ROOT / CANONICAL_MAIL_ARCHIVE_ROOT
    )
    assert default_config(tmp_path, mail_data_root=tmp_path).mail_archive_root == (
        tmp_path / CANONICAL_MAIL_ARCHIVE_ROOT
    )

    config = FullMemoryIngestConfig(
        project_root=tmp_path,
        production_db=tmp_path / "prod" / "customer_timeline.sqlite",
        test_out_root=tmp_path / "test",
    )
    assert config.identity_db == tmp_path / CANONICAL_MAIL_IDENTITY_DB
    assert config.event_jsonl_paths == (
        tmp_path / CANONICAL_MAIL_STAGE2_FULL_EVENTS,
        tmp_path / CANONICAL_MAIL_STAGE2_DELTA_EVENTS,
    )
    assert config.relink_decision_paths == ()

    assert default_identity_db(tmp_path) == tmp_path / CANONICAL_MAIL_IDENTITY_DB
    assert default_corpus_events(tmp_path) == tmp_path / CANONICAL_MAIL_STAGE2_FULL_EVENTS
    assert default_delta_events(tmp_path) == tmp_path / CANONICAL_MAIL_STAGE2_DELTA_EVENTS
    assert build_mail_bridge_parser().parse_args([]).fresh_relink_root is None


def test_mail_bridge_requires_explicit_relink_decisions(tmp_path: Path) -> None:
    args = build_mail_bridge_parser().parse_args(
        ["--data-project-root", str(tmp_path), "--out-root", str(tmp_path / "out")]
    )
    with pytest.raises(ValueError, match="relink decision CSVs are required"):
        run_bridge(args)


def test_runtime_code_has_no_legacy_mail_archive_paths() -> None:
    root = Path(__file__).resolve().parents[1]
    legacy_markers = ("mail_archive_2026-05-12", "mail_archive_2026-06-20")
    offenders = []
    for source_root in (root / "src", root / "scripts"):
        for path in source_root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if any(marker in text for marker in legacy_markers):
                offenders.append(str(path.relative_to(root)))
    assert offenders == []
