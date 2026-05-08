from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping
from unittest.mock import patch

from mango_mvp.productization.mango_office_client import MangoOfficeCredentials
from mango_mvp.productization.mango_recordings import (
    MangoRecordingDownloader,
    parse_location_from_text,
)


class FakePostResponse:
    status_code = 302
    text = ""
    headers = {"Location": "/recordings/download/rec-1.mp3"}


class FakeGetResponse:
    status_code = 200
    text = ""

    def __enter__(self) -> "FakeGetResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def iter_content(self, chunk_size: int):
        yield b"abc"
        yield b""
        yield b"def"


def test_mango_recording_downloader_signs_link_request_and_downloads(tmp_path: Path) -> None:
    posted = []

    def fake_post(url: str, data: Mapping[str, str], timeout: int, allow_redirects: bool) -> FakePostResponse:
        posted.append({"url": url, "data": dict(data), "timeout": timeout, "allow_redirects": allow_redirects})
        return FakePostResponse()

    target = tmp_path / "recording.mp3"
    downloader = MangoRecordingDownloader(
        credentials=MangoOfficeCredentials(api_key="key", api_salt="salt"),
        base_url="https://example.test",
        timeout_sec=7,
        link_retries=0,
    )

    with patch("mango_mvp.productization.mango_recordings.requests.post", side_effect=fake_post), patch(
        "mango_mvp.productization.mango_recordings.requests.get", return_value=FakeGetResponse()
    ) as get:
        size = downloader.download("rec-1", target)

    body = json.dumps({"recording_id": "rec-1", "action": "download"}, ensure_ascii=False, separators=(",", ":"))
    expected_sign = hashlib.sha256(f"key{body}salt".encode("utf-8")).hexdigest()
    assert posted == [
        {
            "url": "https://example.test/vpbx/queries/recording/post/",
            "data": {"vpbx_api_key": "key", "sign": expected_sign, "json": body},
            "timeout": 7,
            "allow_redirects": False,
        }
    ]
    get.assert_called_once_with("https://example.test/recordings/download/rec-1.mp3", stream=True, timeout=7)
    assert size == 6
    assert target.read_bytes() == b"abcdef"


def test_parse_location_from_text_accepts_header_like_body() -> None:
    assert parse_location_from_text("HTTP/1.1 302\nLocation: /x/y.mp3\n") == "/x/y.mp3"
