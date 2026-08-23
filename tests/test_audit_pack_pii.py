from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import make_audit_pack


def test_mask_pii_redacts_phone_email():
    text = (
        "Телефон +7 999 123-45-67, второй 8 (916) 000-11-22, "
        "международный +44 (20) 7946-0958, WhatsApp: 020 7946 0958, "
        "phone: 971500000000, email client@example.com"
    )

    masked = make_audit_pack.mask_pii(text)

    assert "+7 999 123-45-67" not in masked
    assert "8 (916) 000-11-22" not in masked
    assert "client@example.com" not in masked
    assert "+44 (20) 7946-0958" not in masked
    assert "020 7946 0958" not in masked
    assert "971500000000" not in masked
    assert masked.count("[redacted_phone]") == 5
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


def test_removed_public_bot_path_still_requires_semantic_review():
    assert make_audit_pack._semantic_required("A scripts/run_telegram_public_pilot_bots.py")


HEAD = "a" * 40


def _context_repo(tmp_path: Path, monkeypatch) -> tuple[Path, Path, Path]:
    root = tmp_path / "repo"
    task = root / "tasks/_running/TZ.md"
    inventory = root / "audits/_inbox/bootstrap/prebuild_inventory.json"
    owner = root / "scripts/owner.py"
    task.parent.mkdir(parents=True)
    inventory.parent.mkdir(parents=True)
    owner.parent.mkdir(parents=True)
    task.write_text(
        "Feature-ID: feature.context\nProblem-ID: problem.loss\nТелефон +7 999 123-45-67, email client@example.com\n",
        encoding="utf-8",
    )
    inventory.write_text(json.dumps({"selected_owner": {"path": "scripts/owner.py"}}), encoding="utf-8")
    owner.write_text("def owner():\n    return True\n", encoding="utf-8")

    def fake_git(_root: Path, *args: str) -> str:
        command = " ".join(args)
        return HEAD + "\n" if command == "rev-parse HEAD" else "main\n" if command == "rev-parse --abbrev-ref HEAD" else ""

    monkeypatch.setattr(make_audit_pack, "_run_git", fake_git)
    monkeypatch.setattr(make_audit_pack, "_git_required", fake_git)
    return root, task, inventory


def _valid_review(pack: Path) -> str:
    manifest = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))
    return (
        "MODE: READ_ONLY\n"
        f"PACK_DIR: {manifest['pack_path']}\nMANIFEST: {manifest['pack_path']}/manifest.json\n"
        f"NONCE: {manifest['review_nonce']}\n"
        "CONTEXT_READ: task.md, prebuild_inventory.json, git_context.txt, context_files.json, manifest.json\n"
        f"HEAD: {manifest['head']}\nVERDICT: PASS\n" + "Проверка выполнена по исходникам. " * 8
    )


def test_claude_context_pack_is_minimal_masked_hashed_and_manifest_last(tmp_path, monkeypatch):
    root, task, inventory = _context_repo(tmp_path, monkeypatch)
    pack = make_audit_pack.create_claude_context_pack(root, "context", task, inventory)

    assert "[redacted_phone]" in (pack / "task.md").read_text(encoding="utf-8")
    assert "[redacted_email]" in (pack / "task.md").read_text(encoding="utf-8")
    assert (pack / "prebuild_inventory.json").read_bytes() == inventory.read_bytes()
    manifest = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["files"]) == {"task.md", "prebuild_inventory.json", "git_context.txt", "context_files.json", "review_prompt.md"}
    assert manifest["pii_redaction"] == ["ru_phone", "email"]
    assert manifest["secret_handling"] == "listed patterns blocked, not redacted"
    assert f"PACK_DIR: {manifest['pack_path']}" in (pack / "review_prompt.md").read_text(encoding="utf-8")
    assert f"NONCE: {manifest['review_nonce']}" in (pack / "review_prompt.md").read_text(encoding="utf-8")
    assert not make_audit_pack._valid_review_result(
        _valid_review(pack).replace(f"MANIFEST: {manifest['pack_path']}/manifest.json\n", ""),
        manifest["head"], manifest["pack_path"], manifest["review_nonce"],
    )
    assert (pack / "manifest.json").stat().st_mtime_ns >= max(
        item.stat().st_mtime_ns for item in pack.iterdir() if item.name != "manifest.json"
    )
    assert make_audit_pack.verify_claude_context(root, pack) == []


def test_claude_context_verify_detects_pack_and_source_byte_changes(tmp_path, monkeypatch):
    root, task, inventory = _context_repo(tmp_path, monkeypatch)
    pack = make_audit_pack.create_claude_context_pack(root, "context", task, inventory)
    (pack / "task.md").write_text("changed", encoding="utf-8")
    assert any("byte mismatch" in item for item in make_audit_pack.verify_claude_context(root, pack))

    pack = make_audit_pack.create_claude_context_pack(root, "context2", task, inventory)
    (root / "scripts/owner.py").write_text("changed = True\n", encoding="utf-8")
    assert any("source drift" in item for item in make_audit_pack.verify_claude_context(root, pack))


@pytest.mark.parametrize("secret", [
    "OPENAI_API_KEY=sk-proj-abcdefghijklmnop", "Authorization: Bearer abcdefghijklmnop",
    "TELEGRAM_BOT_TOKEN=123456789:AAabcdefghijklmnopqrstuv",
])
def test_claude_context_blocks_secret_before_manifest(tmp_path, monkeypatch, secret):
    root, task, inventory = _context_repo(tmp_path, monkeypatch)
    task.write_text(task.read_text(encoding="utf-8") + secret, encoding="utf-8")
    with pytest.raises(ValueError, match="secret-like"):
        make_audit_pack.create_claude_context_pack(root, "blocked", task, inventory)
    assert not list((root / "audits/_inbox").rglob("manifest.json"))


def test_secret_scan_does_not_join_prose_across_lines():
    make_audit_pack._assert_no_secret("doc", b"requires a separate token:\nnext paragraph is harmless")


def test_claude_context_blocks_pii_in_byte_exact_inventory(tmp_path, monkeypatch):
    root, task, inventory = _context_repo(tmp_path, monkeypatch)
    inventory.write_text('{"note":"client@example.com"}', encoding="utf-8")
    with pytest.raises(ValueError, match="PII-like"):
        make_audit_pack.create_claude_context_pack(root, "blocked_pii", task, inventory)
    assert not list((root / "audits/_inbox").glob("blocked_pii_*"))


def test_claude_context_rejects_forbidden_and_symlink_sources_without_reading(tmp_path, monkeypatch):
    root, task, inventory = _context_repo(tmp_path, monkeypatch)
    forbidden = root / "product_data/private.py"
    forbidden.parent.mkdir()
    forbidden.write_text("private", encoding="utf-8")
    with pytest.raises(ValueError, match="unsafe context"):
        make_audit_pack.create_claude_context_pack(root, "bad", task, inventory, context_files=(forbidden,))

    target = tmp_path / "outside.py"
    target.write_text("outside", encoding="utf-8")
    link = root / "scripts/link.py"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        make_audit_pack.create_claude_context_pack(root, "bad2", task, inventory, context_files=(link,))

    for name in ("client_+79991234567.py", "client_user@example.com.py"):
        pii_path = root / "scripts" / name
        pii_path.write_text("safe = True\n", encoding="utf-8")
        with pytest.raises(ValueError, match="path contains PII-like"):
            make_audit_pack.create_claude_context_pack(root, "bad_pii_path", task, inventory, context_files=(pii_path,))


def test_claude_context_receipt_comes_from_restricted_cli_and_deduplicates(tmp_path, monkeypatch):
    root, task, inventory = _context_repo(tmp_path, monkeypatch)
    pack = make_audit_pack.create_claude_context_pack(root, "context", task, inventory)
    binary = tmp_path / "claude"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    binary.chmod(0o700)

    def fake_run(command, **_kwargs):
        assert command[command.index("--permission-mode") + 1] == "plan"
        assert command[command.index("--tools") + 1] == "Read,Glob,Grep"
        session = command[command.index("--session-id") + 1]
        assert f"PACK_DIR: {pack.relative_to(root)}" in command[-1]
        review = _valid_review(pack)
        output = json.dumps({"session_id": session, "result": review})
        return make_audit_pack.subprocess.CompletedProcess(command, 0, output, "")

    monkeypatch.setattr(make_audit_pack.subprocess, "run", fake_run)
    receipt = make_audit_pack.run_claude_review(root, pack, claude_bin=binary)

    assert make_audit_pack.verify_claude_context(
        root, pack, receipt, expected_task=task, expected_inventory=inventory,
    ) == []
    copied_receipt = pack.with_name("forged_receipt.json")
    copied_receipt.write_bytes(receipt.read_bytes())
    assert "receipt path is not canonical" in make_audit_pack.verify_claude_context(root, pack, copied_receipt)
    with pytest.raises(ValueError, match="duplicate"):
        make_audit_pack.run_claude_review(root, pack, claude_bin=binary)
    stored = pack.with_name(pack.name + "_claude_cli.json")
    stored.write_bytes(stored.read_bytes() + b"\n")
    assert any("Claude output" in item for item in make_audit_pack.verify_claude_context(root, pack, receipt))


def test_claude_context_surface_detects_new_untracked_code(tmp_path, monkeypatch):
    root, task, inventory = _context_repo(tmp_path, monkeypatch)
    (root / "scripts/new.py").write_text("new = True\n", encoding="utf-8")

    def dynamic_git(_root: Path, *args: str) -> str:
        command = " ".join(args)
        if command == "rev-parse HEAD":
            return HEAD + "\n"
        if command == "rev-parse --abbrev-ref HEAD":
            return "main\n"
        if command == "status --porcelain --untracked-files=all":
            return "?? scripts/new.py\n"
        return ""

    monkeypatch.setattr(make_audit_pack, "_git_required", dynamic_git)
    pack = make_audit_pack.create_claude_context_pack(root, "context", task, inventory)
    context = json.loads((pack / "context_files.json").read_text(encoding="utf-8"))["files"]
    assert "scripts/new.py" in context
    (root / "scripts/new.py").write_text("new = False\n", encoding="utf-8")
    errors = make_audit_pack.verify_claude_context(root, pack)
    assert any("source drift" in item for item in errors)
    assert any("code surface" in item for item in errors)


def test_claude_context_blocks_env_and_hides_blocked_status_paths(tmp_path, monkeypatch):
    root, task, inventory = _context_repo(tmp_path, monkeypatch)
    env_file = root / "scripts/.env.example"
    env_file.write_text("OPENAI_API_KEY=<placeholder>\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unsafe context"):
        make_audit_pack.create_claude_context_pack(root, "blocked", task, inventory, context_files=(env_file,))

    nested_env = root / "scripts/.env/config.py"
    nested_env.parent.mkdir()
    nested_env.write_text("VALUE = 'not-secret'\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unsafe context"):
        make_audit_pack.create_claude_context_pack(root, "blocked_nested", task, inventory, context_files=(nested_env,))

    original = make_audit_pack._git_required
    monkeypatch.setattr(
        make_audit_pack, "_git_required",
        lambda repo, *args: "?? product_data/client_ivan.md\n?? scripts/.env/config.py\n"
        if " ".join(args) == "status --porcelain --untracked-files=all" else original(repo, *args),
    )
    pack = make_audit_pack.create_claude_context_pack(root, "safe_status", task, inventory)
    context = (pack / "git_context.txt").read_text(encoding="utf-8")
    assert "client_ivan" not in context
    assert ".env" not in context
    assert "blocked_paths: 2" in context


def test_verifier_rejects_forged_incomplete_pack_and_secret_output(tmp_path, monkeypatch):
    root, task, inventory = _context_repo(tmp_path, monkeypatch)
    pack = make_audit_pack.create_claude_context_pack(root, "context", task, inventory)
    manifest_path = pack / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"].pop("task.md")
    (pack / "task.md").unlink()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert any("file set" in item for item in make_audit_pack.verify_claude_context(root, pack))

    pack = make_audit_pack.create_claude_context_pack(root, "context2", task, inventory)
    binary = tmp_path / "claude"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    binary.chmod(0o700)
    def fake_run(command, **_kwargs):
        session = command[command.index("--session-id") + 1]
        review = _valid_review(pack)
        return make_audit_pack.subprocess.CompletedProcess(command, 0, json.dumps({"session_id": session, "result": review}), "")
    monkeypatch.setattr(make_audit_pack.subprocess, "run", fake_run)
    receipt = make_audit_pack.run_claude_review(root, pack, claude_bin=binary)
    receipt_data = json.loads(receipt.read_text(encoding="utf-8"))
    output = root / receipt_data["output_path"]
    forged = json.loads(output.read_text(encoding="utf-8"))
    forged["result"] += "\nOPENAI_API_KEY=sk-proj-abcdefghijklmnop"
    output.write_text(json.dumps(forged), encoding="utf-8")
    receipt_data["output_sha256"] = make_audit_pack._sha(output.read_bytes())
    receipt.write_text(json.dumps(receipt_data), encoding="utf-8")
    assert any("secret-like" in item for item in make_audit_pack.verify_claude_context(root, pack, receipt))


def test_claude_context_dedupe_is_stable_but_manifest_is_not(tmp_path, monkeypatch):
    root, task, inventory = _context_repo(tmp_path, monkeypatch)
    first = make_audit_pack.create_claude_context_pack(root, "one", task, inventory)
    second = make_audit_pack.create_claude_context_pack(root, "two", task, inventory)
    one = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    two = json.loads((second / "manifest.json").read_text(encoding="utf-8"))
    assert one["dedupe_key"] == two["dedupe_key"]
    assert make_audit_pack._sha((first / "manifest.json").read_bytes()) != make_audit_pack._sha((second / "manifest.json").read_bytes())


def test_legacy_claude_entrypoint_is_wrapper_only():
    wrapper = (make_audit_pack.DEFAULT_ROOT / ".claude/skills/audit-pack-generator/scripts/create_audit_pack.py").read_text(encoding="utf-8")
    assert "scripts/make_audit_pack.py" in wrapper
    assert "FILES =" not in wrapper
