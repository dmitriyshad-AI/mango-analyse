from __future__ import annotations

import hashlib
import json
import os
import secrets
import stat
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import requests

from mango_mvp.productization.mango_office_client import (
    DEFAULT_MANGO_BASE_URL,
    MangoOfficeCredentials,
)


class MangoRecordingDownloader:
    def __init__(
        self,
        credentials: MangoOfficeCredentials,
        base_url: str = DEFAULT_MANGO_BASE_URL,
        timeout_sec: int = 60,
        link_retries: int = 8,
        rate_limit_sleep_sec: float = 30.0,
    ) -> None:
        self.credentials = credentials
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec
        self.link_retries = link_retries
        self.rate_limit_sleep_sec = rate_limit_sleep_sec

    def download(self, recording_id: str, target_path: Path) -> int:
        link = self.resolve_download_link(recording_id)
        return download_url(link=link, target_path=target_path, timeout_sec=self.timeout_sec)

    def resolve_download_link(self, recording_id: str) -> str:
        payload = {"recording_id": recording_id, "action": "download"}
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        sign = hashlib.sha256(
            f"{self.credentials.api_key}{body}{self.credentials.api_salt}".encode("utf-8")
        ).hexdigest()

        response = None
        for attempt in range(max(1, self.link_retries) + 1):
            response = requests.post(
                f"{self.base_url}/vpbx/queries/recording/post/",
                data={
                    "vpbx_api_key": self.credentials.api_key,
                    "sign": sign,
                    "json": body,
                },
                timeout=self.timeout_sec,
                allow_redirects=False,
            )
            if response.status_code != 429:
                break
            if attempt >= self.link_retries:
                break
            time.sleep(_retry_wait(response.headers.get("Retry-After"), self.rate_limit_sleep_sec))

        assert response is not None
        if response.status_code >= 400:
            raise RuntimeError(f"recording link HTTP {response.status_code}: {response.text[:300]}")

        location = response.headers.get("Location") or response.headers.get("location")
        if not location:
            location = parse_location_from_text(response.text)
        if not location:
            raise RuntimeError(f"recording link response has no Location header: {response.text[:300]}")
        return urljoin(self.base_url, location.strip())


def download_url(link: str, target_path: Path, timeout_sec: int = 60) -> int:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_name(
        f".{target_path.name}.{os.getpid()}.{secrets.token_hex(8)}.part"
    )
    if os.path.lexists(target_path):
        raise RuntimeError("recording target already exists; refusing to overwrite")
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(tmp_path, flags, 0o600)
    digest = hashlib.sha256()
    size_bytes = 0
    try:
        with requests.get(link, stream=True, timeout=timeout_sec) as response:
            if response.status_code >= 400:
                raise RuntimeError(f"download HTTP {response.status_code}")
            with os.fdopen(descriptor, "wb") as fh:
                descriptor = -1
                if not stat.S_ISREG(os.fstat(fh.fileno()).st_mode):
                    raise RuntimeError("recording temporary target is not a regular file")
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        fh.write(chunk)
                        digest.update(chunk)
                        size_bytes += len(chunk)
                fh.flush()
                os.fsync(fh.fileno())
        if size_bytes <= 0:
            raise RuntimeError("downloaded file is empty")
        content_length = getattr(response, "headers", {}).get("Content-Length")
        if content_length not in (None, "") and int(content_length) != size_bytes:
            raise RuntimeError("downloaded file size does not match Content-Length")
        if _sha256_file(tmp_path) != digest.hexdigest():
            raise RuntimeError("downloaded file checksum changed before publication")
        # link+unlink is an atomic no-overwrite publication on the same volume.
        os.link(tmp_path, target_path, follow_symlinks=False)
        os.unlink(tmp_path)
        _fsync_directory(target_path.parent)
        return size_bytes
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        tmp_path.unlink(missing_ok=True)
        raise


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def parse_location_from_text(text: str) -> Optional[str]:
    for line in text.splitlines():
        if line.lower().startswith("location:"):
            return line.split(":", 1)[1].strip()
    return None


def _retry_wait(raw_retry_after: Optional[str], fallback: float) -> float:
    try:
        return max(1.0, float(raw_retry_after)) if raw_retry_after else max(1.0, fallback)
    except ValueError:
        return max(1.0, fallback)
