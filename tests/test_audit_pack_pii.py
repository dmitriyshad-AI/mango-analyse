from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import make_audit_pack


def test_mask_pii_redacts_phone_email():
    text = "Телефон +7 999 123-45-67, второй 8 (916) 000-11-22, email client@example.com"

    masked = make_audit_pack.mask_pii(text)

    assert "+7 999 123-45-67" not in masked
    assert "8 (916) 000-11-22" not in masked
    assert "client@example.com" not in masked
    assert masked.count("[redacted_phone]") == 2
    assert "[redacted_email]" in masked


def test_create_audit_pack_redacts_inputs_and_writes_manifest_last(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    tests_file = tmp_path / "test_output.txt"
    tests_file.write_text("ok +7 999 123-45-67 user@example.com", encoding="utf-8")
    monkeypatch.setattr(make_audit_pack, "_run_git", lambda *_args: "")
    monkeypatch.setattr(make_audit_pack, "_changed_files", lambda *_args: "src/mango_mvp/channels/foo.py\n")

    pack = make_audit_pack.create_audit_pack(root, "tz131", out_root=tmp_path / "audits", tests_file=tests_file)

    test_output = (pack / "test_output.txt").read_text(encoding="utf-8")
    assert "+7 999 123-45-67" not in test_output
    assert "user@example.com" not in test_output
    assert "[redacted_phone]" in test_output
    assert "[redacted_email]" in test_output
    manifest = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))
    assert "manifest.json" not in manifest["files_written_before_manifest"]
    assert manifest["semantic_required"] is True
    assert (pack / "semantic_review.md").exists()


def test_audit_pack_rejects_stable_runtime_and_codex_outputs(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    with pytest.raises(ValueError, match="stable_runtime"):
        make_audit_pack.create_audit_pack(root, "bad", out_root=root / "stable_runtime" / "audit")
    with pytest.raises(ValueError, match="codex"):
        make_audit_pack.create_audit_pack(root, "bad", out_root=tmp_path / ".codex" / "audit")
