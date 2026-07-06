from __future__ import annotations

import importlib.util
import json
import sqlite3
import subprocess
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


def test_m1_overlay_pii_scan_fails_on_identity_contacts(tmp_path: Path) -> None:
    module = _load_script_module()
    db = tmp_path / "overlay.sqlite"
    with sqlite3.connect(db) as con:
        con.execute("CREATE TABLE bot_context_chunks (chunk_id TEXT, record_json TEXT)")
        con.execute(
            """
            CREATE TABLE customer_identities (
              customer_id TEXT,
              display_name TEXT,
              primary_phone TEXT,
              primary_email TEXT,
              record_json TEXT
            )
            """
        )
        con.execute(
            "INSERT INTO customer_identities VALUES (?, ?, ?, ?, ?)",
            (
                "customer:1",
                "Иван Иванов",
                "+7 916 123-45-67",
                "parent@example.com",
                json.dumps({"display_name": "Иван Иванов", "primary_phone": "+7 916 123-45-67"}, ensure_ascii=False),
            ),
        )

    hits = module.scan_overlay_for_pii(db)

    assert "customer:1:identity_display_name" in hits
    assert "customer:1:identity_primary_phone" in hits
    assert "customer:1:identity_primary_email" in hits


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


def test_create_git_bundle_records_verified_bundle(monkeypatch, tmp_path: Path) -> None:
    module = _load_script_module()
    calls: list[list[str]] = []

    def fake_run(cmd, text, capture_output, check):
        calls.append(list(cmd))
        if cmd == ["git", "status", "--porcelain"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd[:3] == ["git", "bundle", "create"]:
            Path(cmd[3]).write_text("bundle-bytes", encoding="utf-8")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd[:3] == ["git", "bundle", "verify"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="The bundle is okay\n", stderr="")
        raise AssertionError(cmd)

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    report = module.create_git_bundle(tmp_path, base_ref="94ee1fe")

    assert calls[0] == ["git", "status", "--porcelain"]
    assert calls[1] == ["git", "bundle", "create", str(tmp_path / "email_timeline_20260706_from94ee1fe.bundle"), "94ee1fe..HEAD"]
    assert calls[2] == ["git", "bundle", "verify", str(tmp_path / "email_timeline_20260706_from94ee1fe.bundle")]
    assert report["git_bundle_verify_rc"] == 0
    assert report["git_bundle_verify_output"] == "The bundle is okay"
    assert Path(report["git_bundle_path"]).exists()
    assert Path(report["git_bundle_verify_path"]).read_text(encoding="utf-8") == "The bundle is okay\n"


def test_create_git_bundle_fails_on_dirty_worktree(monkeypatch, tmp_path: Path) -> None:
    module = _load_script_module()

    def fake_run(cmd, text, capture_output, check):
        if cmd == ["git", "status", "--porcelain"]:
            return subprocess.CompletedProcess(cmd, 0, stdout=" M file.py\n", stderr="")
        raise AssertionError(cmd)

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    try:
        module.create_git_bundle(tmp_path, base_ref="94ee1fe")
    except RuntimeError as exc:
        assert "clean worktree" in str(exc)
    else:
        raise AssertionError("dirty worktree must block M1 git bundle")


def test_memory_overlay_v3_1_copies_base_chunks_without_text_rewrite_and_redacts_identity(tmp_path: Path) -> None:
    module = _load_script_module()
    base_db = tmp_path / "base_v3.sqlite"
    overlay_db = tmp_path / "memory_shadow_overlay_v3_1.sqlite"
    with sqlite3.connect(base_db) as con:
        con.execute(
            """
            CREATE TABLE customer_identities (
              customer_id TEXT PRIMARY KEY,
              tenant_id TEXT,
              display_name TEXT,
              primary_phone TEXT,
              primary_email TEXT,
              record_json TEXT
            )
            """
        )
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
        con.execute(
            "INSERT INTO customer_identities VALUES ('customer:1', 'foton', 'Иван Иванов', '+7 916 123-45-67', NULL, ?)",
            (json.dumps({"display_name": "Иван Иванов", "primary_phone": "+7 916 123-45-67"}, ensure_ascii=False),),
        )
        rows = [
            ("c1", "customer_timeline_bot_safe_summary", "bot_safe_summary", "Фотон: безопасная сводка."),
            ("c2", "mail_archive_stage2", "email_message", "Фотон: письмо, [контактные данные у менеджера]."),
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

    report = module.create_memory_overlay_db(base_db, overlay_db, ["customer:1"], base_overlay_db=base_db)

    assert overlay_db.name == "memory_shadow_overlay_v3_1.sqlite"
    assert report["overlay_version"] == "v3.1"
    assert report["dedup_policy"] == "source_identity_only_no_text_normalization"
    assert report["bot_context_chunks"] == 4
    assert report["field_level_diff"]["chunk_field_changes"] == []
    assert report["identity_pii_redaction"]["redacted_rows"] == 1
    with sqlite3.connect(overlay_db) as con:
        raw = con.execute("SELECT record_json FROM bot_context_chunks WHERE chunk_id = 'c2'").fetchone()[0]
        identity = con.execute("SELECT display_name, primary_phone, record_json FROM customer_identities").fetchone()
    assert raw == json.dumps({"text": "Фотон: письмо, [контактные данные у менеджера].", "summary": "Фотон: письмо, [контактные данные у менеджера]."}, ensure_ascii=False)
    assert "контактные данные у менеджера" in raw
    assert identity[0] is None
    assert identity[1] is None
    assert json.loads(identity[2])["display_name"] is None


def test_memory_overlay_v3_1_dedup_uses_source_identity_not_text_similarity(tmp_path: Path) -> None:
    module = _load_script_module()
    source_db = tmp_path / "base_v3.sqlite"
    overlay_db = tmp_path / "memory_shadow_overlay_v3_1.sqlite"
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
            ("same-1", "mail_archive_stage2", "email_message", "sha-1", "Фотон: письмо шаблон."),
            ("same-2", "mail_archive_stage2", "email_message", "sha-1", "Фотон: письмо шаблон."),
            ("similar-other", "mail_archive_stage2", "email_message", "sha-2", "Фотон: письмо шаблон."),
        ]
        for chunk_id, source_system, chunk_type, message_sha, text in rows:
            con.execute(
                "INSERT INTO bot_context_chunks VALUES (?, ?, ?, ?, 1, 0, NULL, '2026-07-01', '2026-07-01', 0, ?)",
                (
                    chunk_id,
                    "customer:1",
                    source_system,
                    chunk_type,
                    json.dumps({"text": text, "summary": text, "message_sha256": message_sha}, ensure_ascii=False),
                ),
            )

    report = module.create_memory_overlay_db(source_db, overlay_db, ["customer:1"], base_overlay_db=source_db)

    assert report["bot_context_chunks"] == 2
    assert [item["chunk_id"] for item in report["dedup_removed"]] == ["same-2"]


def test_expected_memory_hits_counts_brand_visible_chunks(tmp_path: Path) -> None:
    module = _load_script_module()
    overlay_db = tmp_path / "memory_shadow_overlay_v3_1.sqlite"
    with sqlite3.connect(overlay_db) as con:
        con.execute(
            """
            CREATE TABLE bot_context_chunks (
              chunk_id TEXT PRIMARY KEY,
              customer_id TEXT,
              source_system TEXT,
              chunk_type TEXT,
              record_json TEXT
            )
            """
        )
        con.execute(
            "INSERT INTO bot_context_chunks VALUES (?, ?, ?, ?, ?)",
            (
                "visible",
                "customer:1",
                "customer_timeline_bot_safe_summary",
                "bot_safe_summary",
                json.dumps({"text": "Фотон: память.", "relevance_tags": ["bot_safe", "structured", "foton"]}, ensure_ascii=False),
            ),
        )
        con.execute(
            "INSERT INTO bot_context_chunks VALUES (?, ?, ?, ?, ?)",
            (
                "foreign",
                "customer:1",
                "customer_timeline_bot_safe_summary",
                "bot_safe_summary",
                json.dumps({"text": "УНПК: память.", "relevance_tags": ["bot_safe", "structured", "unpk"]}, ensure_ascii=False),
            ),
        )

    report = module.build_expected_memory_hits(
        overlay_db,
        [{"type": "persona", "dialog_id": "d1", "customer_id": "customer:1", "brand": "foton", "category": "memory_rich"}],
    )

    assert report["with_memory"] == 1
    assert report["rows"][0]["expected_visible_chunks"] == 1
    assert report["gate_passed"] is True


def test_expected_memory_hits_flags_required_memory_without_visible_chunks(tmp_path: Path) -> None:
    module = _load_script_module()
    overlay_db = tmp_path / "memory_shadow_overlay_v3_1.sqlite"
    with sqlite3.connect(overlay_db) as con:
        con.execute(
            """
            CREATE TABLE bot_context_chunks (
              chunk_id TEXT PRIMARY KEY,
              customer_id TEXT,
              source_system TEXT,
              chunk_type TEXT,
              record_json TEXT
            )
            """
        )

    report = module.build_expected_memory_hits(
        overlay_db,
        [{"type": "persona", "dialog_id": "d1", "customer_id": "customer:1", "brand": "foton", "category": "memory_rich"}],
    )

    assert report["gate_passed"] is False
    assert report["rows"][0]["status"] == "violation_expected_memory_missing"
    assert report["violations"][0]["reason"] == "requires_memory_but_expected_visible_chunks_zero"


def test_email_quality_review_sample_prioritizes_control_cases() -> None:
    module = _load_script_module()
    rows = [
        {
            "review_id": "regular",
            "message_sha256": "sha-regular",
            "stratum": "other|foton|short",
            "summary_text": "Обычная выжимка.",
            "summary_payload": {"summary": "Обычная выжимка."},
            "full_clean_text_chars": 120,
        },
        {
            "review_id": "leak",
            "message_sha256": "sha-leak",
            "stratum": "other|foton|short",
            "summary_text": "brand_source=content",
            "summary_payload": {"summary": "brand_source=content"},
            "full_clean_text_chars": 120,
        },
        {
            "review_id": "money",
            "message_sha256": "sha-money",
            "stratum": "payment|foton|medium",
            "summary_text": "Оплата 50 000 руб.",
            "summary_payload": {"summary": "Оплата 50 000 руб."},
            "amount_rub": 50_000,
            "event_type_detail": "payment",
            "full_clean_text_chars": 1200,
        },
    ]

    sample = module.select_email_quality_review_sample(rows, limit=3)

    assert [row["message_sha256"] for row in sample[:2]] == ["sha-leak", "sha-money"]
    assert "internal_marker_leak_candidate" in sample[0]["control_reasons"]
    assert "money_case" in sample[1]["control_reasons"]


def test_email_quality_candidates_include_review_needed_and_sanitized_cache_rows(tmp_path: Path) -> None:
    module = _load_script_module()
    db = tmp_path / "staging.sqlite"
    with sqlite3.connect(db) as con:
        con.execute(
            """
            CREATE TABLE a2v3_mail_event_facts (
              event_id TEXT,
              tenant_id TEXT,
              customer_id TEXT,
              message_sha256 TEXT,
              event_type_detail TEXT,
              money_direction TEXT,
              amount_kind TEXT,
              amount_rub INTEGER,
              amount_uncertain INTEGER,
              email_brand TEXT,
              memory_status TEXT,
              client_safe INTEGER
            )
            """
        )
        con.execute(
            """
            CREATE TABLE timeline_events (
              event_id TEXT PRIMARY KEY,
              event_at TEXT,
              direction TEXT,
              subject TEXT,
              summary TEXT,
              record_json TEXT
            )
            """
        )
        con.execute(
            """
            CREATE TABLE email_summary_cache_v1 (
              message_sha256 TEXT,
              source_kind TEXT,
              summary_text TEXT,
              summary_payload_json TEXT
            )
            """
        )
        for idx, source_kind in enumerate(("llm", "llm_review_needed", "sanitized"), start=1):
            event_id = f"event-{idx}"
            sha = f"sha-{idx}"
            con.execute(
                "INSERT INTO a2v3_mail_event_facts VALUES (?, 'foton', 'customer:1', ?, 'other', 'none', NULL, NULL, 0, 'foton', 'usable_memory', 1)",
                (event_id, sha),
            )
            con.execute(
                "INSERT INTO timeline_events VALUES (?, '2026-07-01', 'inbound', 'Тема', 'Summary', ?)",
                (
                    event_id,
                    json.dumps({"record": {"full_clean_text": "Полный текст письма."}}, ensure_ascii=False),
                ),
            )
            con.execute(
                "INSERT INTO email_summary_cache_v1 VALUES (?, ?, ?, ?)",
                (
                    sha,
                    source_kind,
                    "Summary",
                    json.dumps({"summary": "Summary", "summary_review_needed": source_kind == "llm_review_needed"}, ensure_ascii=False),
                ),
            )

    rows = module.load_email_summary_candidates(db)

    assert {row["source_kind"] for row in rows} == {"llm", "llm_review_needed", "sanitized"}


def test_email_quality_prompt_mentions_subject_and_new_failure_classes() -> None:
    module = _load_script_module()

    prompt = module.EMAIL_QUALITY_PROMPT

    assert "`subject`" in prompt
    assert "sample_sha256" in prompt
    assert "missing_business_specifics" in prompt
    assert "internal_leak" in prompt
    assert "false_hidden" in prompt
