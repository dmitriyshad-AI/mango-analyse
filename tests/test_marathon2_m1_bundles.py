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


def test_memory_overlay_v3_includes_opened_rich_chunks(tmp_path: Path) -> None:
    module = _load_script_module()
    source_db = tmp_path / "source.sqlite"
    overlay_db = tmp_path / "memory_shadow_overlay_v3.sqlite"
    with sqlite3.connect(source_db) as con:
        con.execute("CREATE TABLE customer_identities (customer_id TEXT PRIMARY KEY, tenant_id TEXT)")
        con.execute(
            """
            CREATE TABLE bot_context_chunks (
              chunk_id TEXT PRIMARY KEY,
              customer_id TEXT,
              source_system TEXT,
              chunk_type TEXT,
              allowed_for_bot INTEGER,
              requires_manager_review INTEGER,
              superseded_by TEXT,
              event_at TEXT,
              created_at TEXT,
              ordinal INTEGER,
              record_json TEXT
            )
            """
        )
        con.execute("INSERT INTO customer_identities VALUES ('customer:1', 'foton')")
        rows = [
            ("c1", "customer_timeline_bot_safe_summary", "bot_safe_summary", "Фотон: безопасная сводка."),
            ("c2", "mail_archive_stage2", "email_message", "Фотон: письмо, телефон +7 916 123-45-67, email parent@example.com."),
            ("c3", "telegram_history", "channel_message", "Фотон: открытый telegram без контактов."),
            ("c4", "telegram_history", "channel_message", "Фотон: закрытый telegram."),
        ]
        for chunk_id, source_system, chunk_type, text in rows:
            allowed = 0 if chunk_id == "c4" else 1
            review = 1 if chunk_id == "c4" else 0
            con.execute(
                "INSERT INTO bot_context_chunks VALUES (?, ?, ?, ?, ?, ?, NULL, '2026-07-01', '2026-07-01', 0, ?)",
                (
                    chunk_id,
                    "customer:1",
                    source_system,
                    chunk_type,
                    allowed,
                    review,
                    json.dumps({"text": text, "summary": text}, ensure_ascii=False),
                ),
            )

    report = module.create_memory_overlay_db(source_db, overlay_db, ["customer:1"])

    assert overlay_db.name == "memory_shadow_overlay_v3.sqlite"
    assert report["bot_context_chunks"] == 3
    assert report["chunk_type_counts"] == {"bot_safe_summary": 1, "channel_message": 1, "email_message": 1}
    assert report["source_system_counts"]["telegram_history"] == 1
    with sqlite3.connect(overlay_db) as con:
        raw = con.execute("SELECT record_json FROM bot_context_chunks WHERE chunk_id = 'c2'").fetchone()[0]
    assert "+7 916 123-45-67" not in raw
    assert "parent@example.com" not in raw
    assert "контактные данные у менеджера" in raw
