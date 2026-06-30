import json
import subprocess
from pathlib import Path

from scripts import build_job_manifest


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Test")
    (repo / "product_data" / "telegram_dynamic_test_sets").mkdir(parents=True)
    (repo / "product_data" / "knowledge_base" / "kb").mkdir(parents=True)
    (repo / "product_data" / "telegram_dynamic_test_sets" / "smoke.jsonl").write_text('{"type":"persona","dialog_id":"d1"}\n', encoding="utf-8")
    (repo / "product_data" / "knowledge_base" / "kb" / "kb_release_v3_snapshot.json").write_text('{"ok":true}\n', encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "init")
    return repo


def test_build_manifest_is_small_and_contains_checkout_contract(tmp_path):
    repo = _repo(tmp_path)
    manifest = build_job_manifest.build_manifest(
        repo=repo,
        set_rel_path="product_data/telegram_dynamic_test_sets/smoke.jsonl",
        snapshot_rel_path="product_data/knowledge_base/kb/kb_release_v3_snapshot.json",
        env_flags={"TELEGRAM_TEST_FLAG": "1"},
        parallel=4,
        max_hours=2,
        run_out_dir="runs/smoke",
        extra_args=("--brand", "all"),
    )

    assert manifest["schema_version"] == build_job_manifest.SCHEMA_VERSION
    assert manifest["commit_sha"] == _git(repo, "rev-parse", "HEAD")
    assert manifest["set_sha256"] == build_job_manifest.sha256_file(repo / manifest["set_rel_path"])
    assert manifest["snapshot_sha256"] == build_job_manifest.sha256_file(repo / manifest["snapshot_rel_path"])
    assert manifest["env_flags"] == {"TELEGRAM_TEST_FLAG": "1"}
    assert "git checkout " + manifest["commit_sha"] in manifest["m1_manual_procedure"][1]
    assert "--snapshot" in manifest["run_cmd"]
    assert "TELEGRAM_TEST_FLAG=1" in manifest["run_cmd"]
    assert len(json.dumps(manifest, ensure_ascii=False).encode("utf-8")) < 8_192


def test_write_manifest_uses_job_sha_filename(tmp_path):
    repo = _repo(tmp_path)
    manifest = build_job_manifest.build_manifest(
        repo=repo,
        set_rel_path="product_data/telegram_dynamic_test_sets/smoke.jsonl",
        snapshot_rel_path="product_data/knowledge_base/kb/kb_release_v3_snapshot.json",
        env_flags={},
        parallel=1,
        max_hours=1,
    )

    path = build_job_manifest.write_manifest(tmp_path / "jobs", manifest)

    assert path.name == f"job_{manifest['commit_sha'][:12]}.json"
    assert json.loads(path.read_text(encoding="utf-8"))["commit_sha"] == manifest["commit_sha"]


def test_manifest_rejects_untracked_set_and_secret_env(tmp_path):
    repo = _repo(tmp_path)
    (repo / "untracked.jsonl").write_text("{}\n", encoding="utf-8")

    try:
        build_job_manifest.build_manifest(
            repo=repo,
            set_rel_path="untracked.jsonl",
            snapshot_rel_path="product_data/knowledge_base/kb/kb_release_v3_snapshot.json",
            env_flags={},
            parallel=1,
            max_hours=1,
        )
    except build_job_manifest.ManifestError as exc:
        assert "not tracked" in str(exc)
    else:
        raise AssertionError("untracked set must be rejected")

    try:
        build_job_manifest.parse_env_flags(["OPENAI_API_KEY=secret"])
    except build_job_manifest.ManifestError as exc:
        assert "not allowed" in str(exc)
    else:
        raise AssertionError("secret env key must be rejected")


def test_manifest_rejects_absolute_and_parent_paths(tmp_path):
    repo = _repo(tmp_path)

    for bad in ("/tmp/set.jsonl", "../set.jsonl"):
        try:
            build_job_manifest.normalize_rel_path(repo, bad, label="set_rel_path")
        except build_job_manifest.ManifestError:
            pass
        else:
            raise AssertionError(f"bad path accepted: {bad}")
