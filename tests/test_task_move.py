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


def test_done_external_tz_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    source = tmp_path / "external.md"
    source.write_text("Ветка: main\n", encoding="utf-8")

    with pytest.raises(ValueError, match="только для ТЗ внутри tasks"):
        task_move.move_task(root, str(source), "done")

    assert source.exists()
