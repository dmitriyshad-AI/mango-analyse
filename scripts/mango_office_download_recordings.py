#!/usr/bin/env python3
"""Download Mango Office recordings from a missing-call report.

This script is intentionally isolated from the current runtime pipeline. It does
not write DBs, does not start ASR/R+A and does not write to CRM.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence
from urllib.parse import urljoin

import requests

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


DEFAULT_BASE_URL = "https://app.mango-office.ru"
DEFAULT_REPORT = "/tmp/mango_missing_vs_audio_20260507_fuzzy.json"
DEFAULT_OUT_DIR = "product_data/legacy_mango_download_recordings/recordings"
DEFAULT_MANIFEST = "product_data/legacy_mango_download_recordings/manifest.jsonl"


def main(argv: Optional[Sequence[str]] = None) -> int:
    load_env_file()
    args = parse_args(argv)
    api_key = args.api_key or os.getenv("MANGO_OFFICE_API_KEY")
    api_salt = args.api_salt or os.getenv("MANGO_OFFICE_API_SALT")
    if not api_key or not api_salt:
        print("MANGO_OFFICE_API_KEY and MANGO_OFFICE_API_SALT are required", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    rows = load_recording_rows(Path(args.missing_report))
    if args.limit is not None:
        rows = rows[: args.limit]

    downloaded = 0
    skipped = 0
    failed = 0
    for index, row in enumerate(rows, 1):
        result = process_row(
            row=row,
            api_key=api_key,
            api_salt=api_salt,
            base_url=args.base_url,
            out_dir=out_dir,
            timeout_sec=args.timeout_sec,
            link_retries=args.link_retries,
            rate_limit_sleep_sec=args.rate_limit_sleep_sec,
        )
        append_manifest(manifest_path, result)
        status = result["status"]
        if status == "downloaded":
            downloaded += 1
        elif status == "skipped_exists":
            skipped += 1
        else:
            failed += 1
        print(
            f"{index}/{len(rows)} {status} {result.get('filename', '')} "
            f"{result.get('error', '')}",
            flush=True,
        )
        if args.sleep_sec > 0:
            time.sleep(args.sleep_sec)

    print(
        json.dumps(
            {
                "total": len(rows),
                "downloaded": downloaded,
                "skipped_exists": skipped,
                "failed": failed,
                "out_dir": str(out_dir),
                "manifest": str(manifest_path),
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if failed == 0 else 1


def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Mango Office recordings from a missing-call report.")
    parser.add_argument("--missing-report", default=DEFAULT_REPORT)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--base-url", default=os.getenv("MANGO_OFFICE_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--api-key")
    parser.add_argument("--api-salt")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sleep-sec", type=float, default=1.5)
    parser.add_argument("--link-retries", type=int, default=6)
    parser.add_argument("--rate-limit-sleep-sec", type=float, default=30.0)
    parser.add_argument("--timeout-sec", type=int, default=60)
    return parser.parse_args(argv)


def load_env_file() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    load_dotenv()


def load_recording_rows(report_path: Path) -> list[Mapping[str, Any]]:
    data = json.loads(report_path.read_text(encoding="utf-8"))
    rows = []
    for row in data.get("missing", []):
        if row.get("recording_ref"):
            rows.append(row)
    return rows


def process_row(
    row: Mapping[str, Any],
    api_key: str,
    api_salt: str,
    base_url: str,
    out_dir: Path,
    timeout_sec: int,
    link_retries: int,
    rate_limit_sleep_sec: float,
) -> Mapping[str, Any]:
    filename = build_filename(row)
    target_path = out_dir / filename
    base_result = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "status": "",
        "filename": filename,
        "path": str(target_path),
        "started_at_msk": row.get("started_at_msk"),
        "provider_call_id": row.get("provider_call_id"),
        "event_key": row.get("event_key"),
        "recording_ref": row.get("recording_ref"),
    }
    if target_path.exists() and target_path.stat().st_size > 0:
        return {**base_result, "status": "skipped_exists", "size_bytes": target_path.stat().st_size}

    try:
        link = resolve_recording_download_link(
            recording_id=str(row["recording_ref"]),
            api_key=api_key,
            api_salt=api_salt,
            base_url=base_url,
            timeout_sec=timeout_sec,
            retries=link_retries,
            rate_limit_sleep_sec=rate_limit_sleep_sec,
        )
        size_bytes = download_file(link=link, target_path=target_path, timeout_sec=timeout_sec)
        return {**base_result, "status": "downloaded", "size_bytes": size_bytes}
    except Exception as exc:
        return {**base_result, "status": "failed", "error": str(exc)}


def resolve_recording_download_link(
    recording_id: str,
    api_key: str,
    api_salt: str,
    base_url: str,
    timeout_sec: int,
    retries: int,
    rate_limit_sleep_sec: float,
) -> str:
    payload = {"recording_id": recording_id, "action": "download"}
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    sign = hashlib.sha256(f"{api_key}{body}{api_salt}".encode("utf-8")).hexdigest()
    response = None
    for attempt in range(max(1, retries) + 1):
        response = requests.post(
            f"{base_url.rstrip('/')}/vpbx/queries/recording/post/",
            data={"vpbx_api_key": api_key, "sign": sign, "json": body},
            timeout=timeout_sec,
            allow_redirects=False,
        )
        if response.status_code != 429:
            break
        if attempt >= retries:
            break
        retry_after = response.headers.get("Retry-After")
        try:
            wait_sec = float(retry_after) if retry_after else rate_limit_sleep_sec
        except ValueError:
            wait_sec = rate_limit_sleep_sec
        time.sleep(max(1.0, wait_sec))

    assert response is not None
    if response.status_code >= 400:
        raise RuntimeError(f"recording link HTTP {response.status_code}: {response.text[:300]}")

    location = response.headers.get("Location") or response.headers.get("location")
    if not location:
        location = parse_location_from_text(response.text)
    if not location:
        raise RuntimeError(f"recording link response has no Location header: {response.text[:300]}")
    return urljoin(base_url, location.strip())


def parse_location_from_text(text: str) -> Optional[str]:
    for line in text.splitlines():
        if line.lower().startswith("location:"):
            return line.split(":", 1)[1].strip()
    return None


def download_file(link: str, target_path: Path, timeout_sec: int) -> int:
    tmp_path = target_path.with_suffix(target_path.suffix + ".part")
    with requests.get(link, stream=True, timeout=timeout_sec) as response:
        if response.status_code >= 400:
            raise RuntimeError(f"download HTTP {response.status_code}: {response.text[:300]}")
        with tmp_path.open("wb") as fh:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)
    size_bytes = tmp_path.stat().st_size
    if size_bytes <= 0:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError("downloaded file is empty")
    tmp_path.replace(target_path)
    return size_bytes


def build_filename(row: Mapping[str, Any]) -> str:
    started = str(row.get("started_at_msk") or "unknown").replace(":", "-").replace(" ", "__")
    phone = sanitize_filename_part(str(row.get("client_phone") or "no-phone"))
    call_id = sanitize_filename_part(str(row.get("provider_call_id") or row.get("event_key") or "no-id"))
    return f"{started}__{phone}__mango_{call_id}.mp3"


def sanitize_filename_part(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9А-Яа-яёЁ+_.=-]+", "_", value.strip())
    return cleaned.strip("._")[:120] or "unknown"


def append_manifest(manifest_path: Path, result: Mapping[str, Any]) -> None:
    with manifest_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
