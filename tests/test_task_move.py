from __future__ import annotations

from pathlib import Path

import pytest

from scripts import task_move


def test_take_external_tz_copies_without_deleting_source(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    source = tmp_path / "external" / "TZ.md"
    source.parent.mkdir()
    source.write_text("Ветка: main\n", encoding="utf-8")

    destination = task_move.move_task(root, str(source), "take")

    assert source.read_text(encoding="utf-8") == "Ветка: main\n"
    assert destination == root / "tasks/_running/TZ.md"
    assert "Ветка: main" in destination.read_text(encoding="utf-8")


def test_done_internal_tz_moves_and_removes_source(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    source = root / "tasks/_running/TZ.md"
    source.parent.mkdir(parents=True)
    source.write_text("Ветка: main\n", encoding="utf-8")

    destination = task_move.move_task(root, str(source), "done")

    assert not source.exists()
    assert destination.exists()
    assert "Исход: legacy_unknown" in destination.read_text(encoding="utf-8")


def test_done_external_tz_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    source = tmp_path / "external.md"
    source.write_text("Ветка: main\n", encoding="utf-8")

    with pytest.raises(ValueError, match="только для ТЗ внутри tasks"):
        task_move.move_task(root, str(source), "done")

    assert source.exists()


def test_failed_legacy_task_remains_compatible(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    source = root / "tasks/_running/TZ.md"
    source.parent.mkdir(parents=True)
    source.write_text("Ветка: main\n", encoding="utf-8")

    destination = task_move.move_task(root, str(source), "fail", "legacy failure")

    assert "Исход: legacy_unknown" in destination.read_text(encoding="utf-8")


def test_fields_in_task_body_do_not_count_as_metadata(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    source = root / "tasks/_running/TZ.md"
    source.parent.mkdir(parents=True)
    source.write_text("Problem-ID: problem.test\n\n## Example\nClosure-evidence: not-real\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Closure-evidence"):
        task_move.move_task(root, str(source), "done", outcome="problem_closed")


def test_problem_task_requires_explicit_outcome(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    source = root / "tasks/_running/TZ.md"
    source.parent.mkdir(parents=True)
    source.write_text("Problem-ID: problem.test\n", encoding="utf-8")

    with pytest.raises(ValueError, match="требует --outcome"):
        task_move.move_task(root, str(source), "done")

    assert source.exists()


def test_problem_closed_requires_evidence_and_upserts_single_outcome(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    source = root / "tasks/_running/TZ.md"
    source.parent.mkdir(parents=True)
    source.write_text("Problem-ID: problem.test\nИсход: attempt_complete\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Closure-evidence"):
        task_move.move_task(root, str(source), "done", outcome="problem_closed")

    source.write_text(
        "Problem-ID: problem.test\nИсход: attempt_complete\nClosure-evidence: audits/_inbox/proof\n",
        encoding="utf-8",
    )
    destination = task_move.move_task(root, str(source), "done", outcome="problem_closed")
    text = destination.read_text(encoding="utf-8")
    assert text.count("Исход:") == 1
    assert "Исход: problem_closed" in text


def test_superseded_requires_existing_next_step(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    source = root / "tasks/_running/TZ.md"
    source.parent.mkdir(parents=True)
    source.write_text("Problem-ID: problem.test\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Следующий шаг"):
        task_move.move_task(root, str(source), "done", outcome="superseded")

    source.write_text("Problem-ID: problem.test\nСледующий шаг: tasks/_inbox_codex/NEXT.md\n", encoding="utf-8")
    destination = task_move.move_task(root, str(source), "done", outcome="superseded")
    assert "Исход: superseded" in destination.read_text(encoding="utf-8")
