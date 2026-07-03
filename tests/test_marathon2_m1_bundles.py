from __future__ import annotations

import importlib.util
import json
import sqlite3
from pathlib import Path


def _load_script_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_marathon2_m1_bundles.py"
    spec = importlib.util.spec_from_file_location("build_marathon2_m1_bundles", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_m1_overlay_pii_scan_checks_prompt_text_not_service_metadata(tmp_path: Path) -> None:
    module = _load_script_module()
    db = tmp_path / "overlay.sqlite"
    with sqlite3.connect(db) as con:
        con.execute("CREATE TABLE bot_context_chunks (chunk_id TEXT, record_json TEXT)")
        con.execute(
            "INSERT INTO bot_context_chunks VALUES (?, ?)",
            (
                "bot_context_chunk:aaaaaaaaaaaaaaaa",
                json.dumps(
                    {
                        "chunk_id": "bot_context_chunk:aaaaaaaaaaaaaaaa",
                        "customer_id": "customer:bbbbbbbbbbbbbbbb",
                        "source_ref": "botsafe:customer:bbbbbbbbbbbbbbbb:unpk",
                        "text": "Бренд: УНПК. Следующий шаг: менеджер проверяет актуальность.",
                        "summary": "Без телефонов и email.",
                    },
                    ensure_ascii=False,
                ),
            ),
        )

    assert module.scan_overlay_for_pii(db) == []


def test_m1_overlay_pii_scan_fails_on_prompt_text_contacts(tmp_path: Path) -> None:
    module = _load_script_module()
    db = tmp_path / "overlay.sqlite"
    with sqlite3.connect(db) as con:
        con.execute("CREATE TABLE bot_context_chunks (chunk_id TEXT, record_json TEXT)")
        con.execute(
            "INSERT INTO bot_context_chunks VALUES (?, ?)",
            (
                "bot_context_chunk:cccccccccccccccc",
                json.dumps(
                    {
                        "text": "Позвонить по +7 916 123-45-67",
                        "summary": "Письмо на client@example.com",
                    },
                    ensure_ascii=False,
                ),
            ),
        )

    hits = module.scan_overlay_for_pii(db)

    assert "bot_context_chunk:cccccccccccccccc:phone" in hits
    assert "bot_context_chunk:cccccccccccccccc:email" in hits


def test_memory_shadow_commands_do_not_start_m1_queue(tmp_path: Path) -> None:
    module = _load_script_module()
    text = module.render_memory_shadow_commands(
        micro_path=Path("micro.jsonl"),
        full_path=Path("full.jsonl"),
        overlay_db=Path("overlay.sqlite"),
        snapshot_path=Path("snapshot.json"),
        out_dir=tmp_path,
        parallel=4,
    )

    assert "--execute" not in text
    assert "--streams-ready" not in text
    assert "run_memory_measure_off_on.py" in text
    assert "человек запускает на M1 вручную" in text
