from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _is_git_ignored(path: str) -> bool:
    proc = subprocess.run(
        ["git", "check-ignore", path],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode == 0


def test_whatsapp_raw_archive_is_ignored_by_git() -> None:
    assert _is_git_ignored("all_whatsapp_chats.txt")


def test_customer_profiles_data_directory_is_ignored_by_git() -> None:
    assert _is_git_ignored("product_data/customer_profiles/customer_profiles.sqlite")
    assert _is_git_ignored("product_data/customer_profiles/raw_profile_dump.jsonl")
