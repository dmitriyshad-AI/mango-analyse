from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Dict, Optional, Tuple


AFINFO_DATA_RE = re.compile(
    r"Data format:\s*(?P<channels>\d+)\s+ch,\s*(?P<sample_rate>\d+)\s+Hz,\s*\.(?P<codec>[A-Za-z0-9_+-]+)",
    re.I,
)
AFINFO_DURATION_RE = re.compile(
    r"(?:estimated duration|duration):\s*(?P<duration>[0-9]+(?:\.[0-9]+)?)\s*sec",
    re.I,
)
AFINFO_TYPE_RE = re.compile(r"File type ID:\s*(?P<file_type>\S+)", re.I)


def _empty_audio_meta() -> Dict[str, Optional[float]]:
    return {
        "codec_name": None,
        "sample_rate": None,
        "channels": None,
        "duration_sec": None,
    }


def _merge_meta(
    primary: Dict[str, Optional[float]],
    fallback: Dict[str, Optional[float]],
) -> Dict[str, Optional[float]]:
    merged = dict(primary)
    for key in ("codec_name", "sample_rate", "channels", "duration_sec"):
        if merged.get(key) is None:
            merged[key] = fallback.get(key)
    return merged


def _probe_audio_ffprobe(path: Path) -> Dict[str, Optional[float]]:
    if shutil.which("ffprobe") is None:
        return _empty_audio_meta()
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "stream=codec_name,sample_rate,channels:format=duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return _empty_audio_meta()
    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        return _empty_audio_meta()
    streams = payload.get("streams", [])
    stream = streams[0] if streams else {}
    duration_raw = (payload.get("format") or {}).get("duration")
    try:
        duration_sec = float(duration_raw) if duration_raw is not None else None
    except (TypeError, ValueError):
        duration_sec = None
    try:
        sample_rate = int(stream.get("sample_rate")) if stream.get("sample_rate") else None
    except (TypeError, ValueError):
        sample_rate = None
    try:
        channels = int(stream.get("channels")) if stream.get("channels") else None
    except (TypeError, ValueError):
        channels = None
    return {
        "codec_name": stream.get("codec_name"),
        "sample_rate": sample_rate,
        "channels": channels,
        "duration_sec": duration_sec,
    }


def _probe_audio_afinfo(path: Path) -> Dict[str, Optional[float]]:
    if shutil.which("afinfo") is None:
        return _empty_audio_meta()
    cmd = ["afinfo", str(path)]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return _empty_audio_meta()
    output = result.stdout or ""

    codec_name: Optional[str] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    duration_sec: Optional[float] = None

    data_match = AFINFO_DATA_RE.search(output)
    if data_match:
        codec_name = str(data_match.group("codec")).lower()
        try:
            sample_rate = int(data_match.group("sample_rate"))
        except (TypeError, ValueError):
            sample_rate = None
        try:
            channels = int(data_match.group("channels"))
        except (TypeError, ValueError):
            channels = None

    duration_match = AFINFO_DURATION_RE.search(output)
    if duration_match:
        try:
            duration_sec = float(duration_match.group("duration"))
        except (TypeError, ValueError):
            duration_sec = None

    if codec_name is None:
        type_match = AFINFO_TYPE_RE.search(output)
        if type_match:
            codec_name = str(type_match.group("file_type")).strip().lower()

    return {
        "codec_name": codec_name,
        "sample_rate": sample_rate,
        "channels": channels,
        "duration_sec": duration_sec,
    }


def _probe_audio_wave(path: Path) -> Dict[str, Optional[float]]:
    if path.suffix.lower() != ".wav":
        return _empty_audio_meta()
    try:
        with wave.open(str(path), "rb") as wf:
            channels = int(wf.getnchannels())
            sample_rate = int(wf.getframerate())
            frames = int(wf.getnframes())
            duration_sec = float(frames) / float(sample_rate) if sample_rate > 0 else None
    except (wave.Error, OSError, EOFError):
        return _empty_audio_meta()
    return {
        "codec_name": "pcm_s16le",
        "sample_rate": sample_rate,
        "channels": channels,
        "duration_sec": duration_sec,
    }


def probe_audio(path: Path) -> Dict[str, Optional[float]]:
    # Prefer ffprobe, then fallback to macOS afinfo, then WAV stdlib parser.
    meta = _probe_audio_ffprobe(path)
    if all(meta.get(k) is not None for k in ("sample_rate", "channels", "duration_sec")):
        return meta
    meta = _merge_meta(meta, _probe_audio_afinfo(path))
    if all(meta.get(k) is not None for k in ("sample_rate", "channels", "duration_sec")):
        return meta
    return _merge_meta(meta, _probe_audio_wave(path))


def split_stereo_to_mono(path: Path) -> Optional[Tuple[Path, Path, Path]]:
    if shutil.which("ffmpeg") is None:
        return None

    temp_dir = Path(tempfile.mkdtemp(prefix="mango_mvp_split_"))
    left = temp_dir / "left.wav"
    right = temp_dir / "right.wav"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(path),
        "-filter_complex",
        "[0:a]channelsplit=channel_layout=stereo[left][right]",
        "-map",
        "[left]",
        str(left),
        "-map",
        "[right]",
        str(right),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not left.exists() or not right.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
        return None
    return left, right, temp_dir
