from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import pull_mango_calls_drop_remote as puller
from scripts import receive_mango_calls_drop as receiver

ROOT = Path(__file__).resolve().parents[1]
RSYNC_GATE = ROOT / "scripts" / "mango_calls_readonly_rsync_gate.sh"


def make_package(root: Path, value: str) -> tuple[Path, str]:
    root.mkdir(parents=True)
    db = root / receiver.DB_NAME
    with sqlite3.connect(db) as connection:
        connection.execute("CREATE TABLE evidence(value TEXT NOT NULL)")
        connection.execute("INSERT INTO evidence VALUES (?)", (value,))
    sha = hashlib.sha256(db.read_bytes()).hexdigest()
    (root / receiver.MANIFEST_NAME).write_text(json.dumps({
        "schema_version": receiver.SCHEMA, "status": "ready", "sha256": sha,
        "size_bytes": db.stat().st_size, "quick_check": "ok", "counts": {"total": 1},
    }), encoding="utf-8")
    return root, sha


def ssh_files(root: Path) -> dict[str, Path]:
    identity, known_hosts = root / "identity", root / "known_hosts"
    identity.write_text("identity", encoding="utf-8")
    known_hosts.write_text("known-host", encoding="utf-8")
    identity.chmod(0o600)
    known_hosts.chmod(0o600)
    return {"identity_file": identity, "known_hosts": known_hosts}


def test_receiver_dry_run_does_not_write(tmp_path: Path) -> None:
    package, sha = make_package(tmp_path / "incoming", "one")
    target = tmp_path / "pipeline"
    result = receiver.accept_drop(package, target, sha, execute=False, confirmation="")

    assert result["status"] == "dry_run"
    assert not target.exists()


def test_receiver_accepts_reuses_and_preserves_one_rollback(tmp_path: Path) -> None:
    first, sha1 = make_package(tmp_path / "first", "one")
    second, sha2 = make_package(tmp_path / "second", "two")
    target = tmp_path / "pipeline"

    accepted = receiver.accept_drop(first, target, sha1, execute=True, confirmation=receiver.CONFIRMATION)
    reused = receiver.accept_drop(first, target, sha1, execute=True, confirmation=receiver.CONFIRMATION)
    replaced = receiver.accept_drop(second, target, sha2, execute=True, confirmation=receiver.CONFIRMATION)

    assert accepted["status"] == "accepted" and reused["status"] == "reused"
    assert replaced["status"] == "accepted"
    drop = target / "drop"
    assert receiver.sha256_file(drop / receiver.DB_NAME) == sha2
    assert receiver.sha256_file(drop / "rollback" / receiver.DB_NAME) == sha1
    assert receiver.load_package(drop, sha2, exact=False)[1]["remote_handoff"] is True

    restored = receiver.restore_rollback(target, execute=True, confirmation=receiver.RESTORE_CONFIRMATION)
    assert restored["status"] == "restored"
    assert receiver.sha256_file(drop / receiver.DB_NAME) == sha1
    assert receiver.sha256_file(drop / "rollback" / receiver.DB_NAME) == sha2


def test_receiver_rollback_recovers_torn_db_manifest_pair(tmp_path: Path) -> None:
    first, sha1 = make_package(tmp_path / "first", "one")
    second, sha2 = make_package(tmp_path / "second", "two")
    target = tmp_path / "pipeline"
    receiver.accept_drop(first, target, sha1, execute=True, confirmation=receiver.CONFIRMATION)
    receiver.accept_drop(second, target, sha2, execute=True, confirmation=receiver.CONFIRMATION)
    drop = target / "drop"
    (drop / receiver.MANIFEST_NAME).write_bytes((drop / "rollback" / receiver.MANIFEST_NAME).read_bytes())

    restored = receiver.restore_rollback(target, execute=True, confirmation=receiver.RESTORE_CONFIRMATION)

    assert restored["sha256"] == sha1
    assert receiver.load_package(drop, sha1, exact=False)[1]["remote_handoff"] is True


def test_receiver_restore_stops_before_write_if_reverse_rollback_cannot_be_staged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, sha1 = make_package(tmp_path / "first", "one")
    second, sha2 = make_package(tmp_path / "second", "two")
    target = tmp_path / "pipeline"
    receiver.accept_drop(first, target, sha1, execute=True, confirmation=receiver.CONFIRMATION)
    receiver.accept_drop(second, target, sha2, execute=True, confirmation=receiver.CONFIRMATION)
    original_stage = receiver.stage_package

    def fail_current_stage(db: Path, manifest: Path, directory: Path) -> tuple[Path, Path]:
        if db.parent == target / "drop":
            raise OSError("synthetic staging failure")
        return original_stage(db, manifest, directory)

    monkeypatch.setattr(receiver, "stage_package", fail_current_stage)
    with pytest.raises(OSError, match="synthetic"):
        receiver.restore_rollback(target, execute=True, confirmation=receiver.RESTORE_CONFIRMATION)

    current_db, _ = receiver.load_package(target / "drop", sha2, exact=False)
    assert receiver.sha256_file(current_db) == sha2


def test_receiver_blocks_tamper_unsafe_path_and_confirmation(tmp_path: Path) -> None:
    package, sha = make_package(tmp_path / "incoming", "one")
    (package / receiver.DB_NAME).write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="verification failed"):
        receiver.accept_drop(package, tmp_path / "pipeline", sha, execute=False, confirmation="")
    clean, clean_sha = make_package(tmp_path / "clean", "one")
    with pytest.raises(RuntimeError, match="confirmation"):
        receiver.accept_drop(clean, tmp_path / "pipeline", clean_sha, execute=True, confirmation="wrong")
    with pytest.raises(RuntimeError, match="unsafe"):
        receiver.accept_drop(clean, tmp_path / "Yandex.Disk.localized" / "pipeline", clean_sha,
                             execute=False, confirmation="")
    with pytest.raises(RuntimeError, match="unsafe"):
        receiver.accept_drop(clean, tmp_path / "STABLE_RUNTIME" / "pipeline", clean_sha,
                             execute=False, confirmation="")


def test_receiver_rejects_symlinked_drop_and_serializes_acceptance(tmp_path: Path) -> None:
    package, sha = make_package(tmp_path / "incoming", "one")
    pipeline, forbidden = tmp_path / "pipeline", tmp_path / "Yandex.Disk.localized" / "drop"
    pipeline.mkdir()
    forbidden.mkdir(parents=True)
    (pipeline / "drop").symlink_to(forbidden, target_is_directory=True)
    with pytest.raises(RuntimeError, match="symlink"):
        receiver.accept_drop(package, pipeline, sha, execute=True, confirmation=receiver.CONFIRMATION)
    assert not any(forbidden.iterdir())
    (pipeline / "drop").unlink()
    with receiver.handoff_lock(pipeline):
        with pytest.raises(RuntimeError, match="already running"):
            receiver.accept_drop(package, pipeline, sha, execute=True, confirmation=receiver.CONFIRMATION)


def test_receiver_cleanup_happens_only_after_accept(tmp_path: Path) -> None:
    package, sha = make_package(tmp_path / "incoming", "one")
    receiver.accept_drop(package, tmp_path / "pipeline", sha, execute=True,
                         confirmation=receiver.CONFIRMATION, cleanup=True)
    assert not package.exists()


class FakePullRunner:
    def __init__(self, package: Path, *, change_second_manifest: bool = False):
        self.package, self.change_second_manifest, self.commands, self.manifests = package, change_second_manifest, [], 0

    def __call__(self, command: list[str] | tuple[str, ...]) -> subprocess.CompletedProcess[str]:
        self.commands.append(list(command))
        source, destination = command[-2], Path(command[-1])
        if source.endswith(receiver.MANIFEST_NAME):
            self.manifests += 1
            raw = (self.package / receiver.MANIFEST_NAME).read_bytes()
            destination.write_bytes(raw + (b" " if self.change_second_manifest and self.manifests == 2 else b""))
        else:
            destination.write_bytes((self.package / receiver.DB_NAME).read_bytes())
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")


def test_pull_dry_run_has_no_network_and_execute_accepts_exact_drop(tmp_path: Path) -> None:
    package, sha = make_package(tmp_path / "remote", "one")
    runner = FakePullRunner(package)
    kwargs = {
        "remote_host": "m1-worker", "remote_drop_root": "/Users/test/.mango_local/pipeline/drop",
        "incoming_root": tmp_path / "incoming", "pipeline_root": tmp_path / "pipeline",
        **ssh_files(tmp_path),
    }
    assert puller.pull_drop(**kwargs, execute=False, confirmation="", runner=runner)["status"] == "dry_run"
    assert not runner.commands

    result = puller.pull_drop(**kwargs, execute=True, confirmation=puller.CONFIRMATION, runner=runner)

    assert result["status"] == "accepted" and result["sha256"] == sha and len(runner.commands) == 3
    assert all(command[0] == "/usr/bin/rsync" and "BatchMode=yes" in " ".join(command) for command in runner.commands)
    assert receiver.sha256_file(tmp_path / "pipeline" / "drop" / receiver.DB_NAME) == sha

    second = FakePullRunner(package)
    reused = puller.pull_drop(**kwargs, execute=True, confirmation=puller.CONFIRMATION, runner=second)
    assert reused["status"] == "reused" and len(second.commands) == 1


def test_rsync_command_uses_owner_only_identity_and_known_hosts(tmp_path: Path) -> None:
    identity, known_hosts = tmp_path / "identity", tmp_path / "known_hosts"
    identity.write_text("private placeholder", encoding="utf-8")
    known_hosts.write_text("host placeholder", encoding="utf-8")
    identity.chmod(0o600)
    known_hosts.chmod(0o600)

    command = puller.rsync_command(
        "worker", "/drop/manifest.json", tmp_path / "manifest.json",
        identity_file=identity, known_hosts=known_hosts,
    )

    assert f"-i {identity}" in command[4]
    assert "IdentitiesOnly=yes" in command[4]
    assert f"UserKnownHostsFile={known_hosts}" in command[4]
    identity.chmod(0o644)
    with pytest.raises(RuntimeError, match="owner-only"):
        puller.rsync_command(
            "worker", "/drop/manifest.json", tmp_path / "manifest.json",
            identity_file=identity, known_hosts=known_hosts,
        )
    identity.chmod(0o600)
    symlink = tmp_path / "identity-link"
    symlink.symlink_to(identity)
    with pytest.raises(RuntimeError, match="owner-only"):
        puller.rsync_command(
            "worker", "/drop/manifest.json", tmp_path / "manifest.json",
            identity_file=symlink, known_hosts=known_hosts,
        )


def test_execute_cli_requires_dedicated_ssh_files(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="dedicated SSH"):
        puller.pull_drop(
            remote_host="worker", remote_drop_root="/drop",
            incoming_root=tmp_path / "incoming-library", pipeline_root=tmp_path / "pipeline-library",
            execute=True, confirmation=puller.CONFIRMATION,
        )
    result = subprocess.run(
        [
            sys.executable, str(Path(puller.__file__)),
            "--remote-host", "worker", "--remote-drop-root", "/drop",
            "--incoming-root", str(tmp_path / "incoming"),
            "--pipeline-root", str(tmp_path / "pipeline"),
            "--config", str(tmp_path / "config.json"),
            "--execute", "--confirmation", puller.CONFIRMATION,
        ],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )

    assert result.returncode == 1
    assert json.loads(result.stdout)["stop_reason"] == "pull_exception:RuntimeError"


def test_pull_rejects_confirmation_unsafe_paths_and_manifest_race(tmp_path: Path) -> None:
    package, _ = make_package(tmp_path / "remote", "one")
    kwargs = {
        "remote_host": "m1-worker", "remote_drop_root": "/Users/test/.mango_local/pipeline/drop",
        "incoming_root": tmp_path / "incoming", "pipeline_root": tmp_path / "pipeline",
        **ssh_files(tmp_path),
    }
    with pytest.raises(RuntimeError, match="confirmation"):
        puller.pull_drop(**kwargs, execute=True, confirmation="wrong", runner=FakePullRunner(package))
    with pytest.raises(RuntimeError, match="unsafe"):
        puller.safe_remote_path("/tmp/../escape")
    with pytest.raises(RuntimeError, match="unsafe"):
        puller.safe_remote_path("/tmp/path with spaces")
    with pytest.raises(RuntimeError, match="host is invalid"):
        puller.safe_remote_host("-oProxyCommand")
    with pytest.raises(RuntimeError, match="changed during transfer"):
        puller.pull_drop(**kwargs, execute=True, confirmation=puller.CONFIRMATION,
                         runner=FakePullRunner(package, change_second_manifest=True))
    assert not any((tmp_path / "incoming").iterdir())


def test_pull_rejects_symlink_manifest_before_reading_it(tmp_path: Path) -> None:
    target = tmp_path / "manifest.json"
    target.write_text("{}", encoding="utf-8")
    link = tmp_path / "remote.manifest.json"
    link.symlink_to(target)
    with pytest.raises(RuntimeError, match="regular file"):
        puller.manifest_sha(link)


def test_pull_rejects_symlinked_incoming_before_network(tmp_path: Path) -> None:
    package, _ = make_package(tmp_path / "remote", "one")
    forbidden = tmp_path / "Yandex.Disk.localized"
    forbidden.mkdir()
    incoming = tmp_path / "incoming"
    incoming.symlink_to(forbidden, target_is_directory=True)
    runner = FakePullRunner(package)
    with pytest.raises(RuntimeError, match="symlink"):
        puller.pull_drop(
            remote_host="m1-worker", remote_drop_root="/Users/test/.mango_local/drop",
            incoming_root=incoming, pipeline_root=tmp_path / "pipeline", execute=True,
            confirmation=puller.CONFIRMATION, runner=runner, **ssh_files(tmp_path),
        )
    assert not runner.commands and not any(forbidden.iterdir())


def test_pull_then_process_b_holds_order_and_blocks_non_success(tmp_path: Path) -> None:
    package, _ = make_package(tmp_path / "remote", "one")
    events: list[str] = []

    class OrderedTransfer(FakePullRunner):
        def __call__(self, command: list[str] | tuple[str, ...]) -> subprocess.CompletedProcess[str]:
            events.append("transfer")
            return super().__call__(command)

    def process_ok(command: list[str] | tuple[str, ...]) -> subprocess.CompletedProcess[str]:
        events.append("process_b")
        return subprocess.CompletedProcess(command, 0, stdout='{"status":"ok","stop_reason":""}\n', stderr="")

    result = puller.pull_then_process_b(
        remote_host="m1-worker", remote_drop_root="/Users/test/.mango_local/drop",
        incoming_root=tmp_path / "incoming", pipeline_root=tmp_path / "pipeline",
        config=tmp_path / "config.json", execute=True, confirmation=puller.CONFIRMATION,
        transfer_runner=OrderedTransfer(package), process_runner=process_ok, **ssh_files(tmp_path),
    )
    assert result["process_b_status"] == "ok" and events == ["transfer", "transfer", "transfer", "process_b"]

    other, _ = make_package(tmp_path / "remote-two", "two")
    with pytest.raises(RuntimeError, match="did not complete"):
        puller.pull_then_process_b(
            remote_host="m1-worker", remote_drop_root="/Users/test/.mango_local/drop",
            incoming_root=tmp_path / "incoming", pipeline_root=tmp_path / "pipeline",
            config=tmp_path / "config.json", execute=True, confirmation=puller.CONFIRMATION,
            transfer_runner=FakePullRunner(other), **ssh_files(tmp_path),
            process_runner=lambda command: subprocess.CompletedProcess(
                command, 0, stdout='{"status":"locked","stop_reason":"timeline_writer_locked"}\n', stderr=""),
        )


def test_readonly_rsync_gate_allows_only_two_sender_paths(tmp_path: Path) -> None:
    drop = tmp_path / "drop"
    drop.mkdir()
    (drop / receiver.MANIFEST_NAME).write_text("{}", encoding="utf-8")
    (drop / receiver.DB_NAME).write_bytes(b"sqlite")
    root = str(drop)
    for filename in (receiver.MANIFEST_NAME, receiver.DB_NAME):
        allowed = f"rsync --server --sender -g -l -o -p -D -r -t --dirs . {root}/{filename}"
        result = subprocess.run([str(RSYNC_GATE), root, "--validate-only"],
                                env={**__import__("os").environ, "SSH_ORIGINAL_COMMAND": allowed},
                                text=True, capture_output=True, check=False)
        assert result.returncode == 0 and result.stdout.strip() == "ok"
    for rejected in (
        f"rsync --server -g -l -o -p -D -r -t --dirs . {root}/{receiver.MANIFEST_NAME}",
        f"rsync --server --sender -g -l -o -p -D -r -t --delete . {root}/{receiver.DB_NAME}",
        "cat /Users/test/.mango_secrets/mango.env",
        f"rsync --server --sender -g -l -o -p -D -r -t --dirs . {root}/other.sqlite",
    ):
        blocked = subprocess.run([str(RSYNC_GATE), root, "--validate-only"],
                                 env={**__import__("os").environ, "SSH_ORIGINAL_COMMAND": rejected},
                                 check=False)
        assert blocked.returncode == 126

    (drop / receiver.MANIFEST_NAME).unlink()
    (drop / receiver.MANIFEST_NAME).symlink_to(drop / receiver.DB_NAME)
    symlink_command = f"rsync --server --sender -g -l -o -p -D -r -t --dirs . {root}/{receiver.MANIFEST_NAME}"
    blocked = subprocess.run([str(RSYNC_GATE), root, "--validate-only"],
                             env={**__import__("os").environ, "SSH_ORIGINAL_COMMAND": symlink_command},
                             check=False)
    assert blocked.returncode == 126
