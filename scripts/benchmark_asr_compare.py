#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Tuple

import mlx_whisper
from gigaam import load_model

DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)")
WORD_RE = re.compile(r"[\w'-]+", re.UNICODE)


def run_cmd(cmd: List[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def probe_duration_sec(ffmpeg_bin: str, path: Path) -> float:
    result = run_cmd([ffmpeg_bin, "-hide_banner", "-i", str(path)])
    match = DURATION_RE.search(result.stderr)
    if not match:
        raise RuntimeError(f"Could not read duration for: {path}")
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def split_stereo_to_mono(ffmpeg_bin: str, src: Path, out_dir: Path) -> Tuple[Path, Path]:
    left = out_dir / "left.wav"
    right = out_dir / "right.wav"
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(src),
        "-filter_complex",
        "[0:a]channelsplit=channel_layout=stereo[left][right]",
        "-map",
        "[left]",
        str(left),
        "-map",
        "[right]",
        str(right),
    ]
    result = run_cmd(cmd)
    if result.returncode != 0 or not left.exists() or not right.exists():
        raise RuntimeError(f"Stereo split failed for {src}:\n{result.stderr}")
    return left, right


def cut_wav_prefix(ffmpeg_bin: str, src: Path, dst: Path, seconds: int) -> None:
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(src),
        "-t",
        str(seconds),
        "-acodec",
        "pcm_s16le",
        str(dst),
    ]
    result = run_cmd(cmd)
    if result.returncode != 0 or not dst.exists():
        raise RuntimeError(f"Failed to cut wav prefix: {src} -> {dst}\n{result.stderr}")


def segment_wav(ffmpeg_bin: str, src: Path, out_dir: Path, segment_sec: int) -> List[Path]:
    pattern = out_dir / "chunk_%03d.wav"
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(src),
        "-f",
        "segment",
        "-segment_time",
        str(segment_sec),
        "-acodec",
        "pcm_s16le",
        str(pattern),
    ]
    result = run_cmd(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to segment wav {src}:\n{result.stderr}")
    chunks = sorted(out_dir.glob("chunk_*.wav"))
    if not chunks:
        raise RuntimeError(f"No chunks generated for: {src}")
    return chunks


def text_metrics(text: str) -> Dict[str, Any]:
    words = [w.lower() for w in WORD_RE.findall(text)]
    max_run = 0
    prev = None
    cur = 0
    for word in words:
        if word == prev:
            cur += 1
        else:
            cur = 1
            prev = word
        if cur > max_run:
            max_run = cur
    return {
        "chars": len(text),
        "words": len(words),
        "max_word_run": max_run,
    }


def transcribe_whisper_channel(path: Path, model: str, language: str) -> str:
    result = mlx_whisper.transcribe(
        str(path),
        path_or_hf_repo=model,
        language=language,
        condition_on_previous_text=False,
    )
    text = result.get("text") if isinstance(result, dict) else None
    if not text:
        return ""
    return " ".join(str(text).split())


def transcribe_gigaam_channel(
    ffmpeg_bin: str,
    path: Path,
    gigaam_model: Any,
    segment_sec: int,
) -> str:
    with tempfile.TemporaryDirectory(prefix="gigaam_chunks_") as td:
        chunks = segment_wav(ffmpeg_bin, path, Path(td), segment_sec=segment_sec)
        parts: List[str] = []
        for chunk in chunks:
            text = str(gigaam_model.transcribe(str(chunk))).strip()
            if text:
                parts.append(" ".join(text.split()))
    return " ".join(parts)


def benchmark_whisper(
    ffmpeg_bin: str,
    files: List[Path],
    model: str,
    language: str,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="whisper_warmup_") as td:
        left, _ = split_stereo_to_mono(ffmpeg_bin, files[0], Path(td))
        warm = Path(td) / "warmup.wav"
        cut_wav_prefix(ffmpeg_bin, left, warm, seconds=5)
        _ = transcribe_whisper_channel(warm, model=model, language=language)

    for file_path in files:
        duration_sec = probe_duration_sec(ffmpeg_bin, file_path)
        with tempfile.TemporaryDirectory(prefix="whisper_split_") as td:
            left, right = split_stereo_to_mono(ffmpeg_bin, file_path, Path(td))
            started = time.perf_counter()
            manager = transcribe_whisper_channel(left, model=model, language=language)
            client = transcribe_whisper_channel(right, model=model, language=language)
            elapsed = time.perf_counter() - started

        combined = f"MANAGER:\n{manager}\n\nCLIENT:\n{client}"
        metrics = text_metrics(combined)
        rows.append(
            {
                "file": str(file_path),
                "duration_sec": round(duration_sec, 3),
                "elapsed_sec": round(elapsed, 3),
                "rtf": round(elapsed / duration_sec, 4),
                **metrics,
            }
        )

    return {
        "model": f"mlx-whisper:{model}",
        "rows": rows,
        "avg_elapsed_sec": round(mean(r["elapsed_sec"] for r in rows), 3),
        "avg_rtf": round(mean(r["rtf"] for r in rows), 4),
    }


def benchmark_gigaam(
    ffmpeg_bin: str,
    files: List[Path],
    model_name: str,
    segment_sec: int,
) -> Dict[str, Any]:
    load_started = time.perf_counter()
    model = load_model(model_name, device="cpu", fp16_encoder=False)
    load_elapsed = time.perf_counter() - load_started

    rows: List[Dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="gigaam_warmup_") as td:
        left, _ = split_stereo_to_mono(ffmpeg_bin, files[0], Path(td))
        warm = Path(td) / "warmup.wav"
        cut_wav_prefix(ffmpeg_bin, left, warm, seconds=5)
        _ = str(model.transcribe(str(warm))).strip()

    for file_path in files:
        duration_sec = probe_duration_sec(ffmpeg_bin, file_path)
        with tempfile.TemporaryDirectory(prefix="gigaam_split_") as td:
            left, right = split_stereo_to_mono(ffmpeg_bin, file_path, Path(td))
            started = time.perf_counter()
            manager = transcribe_gigaam_channel(
                ffmpeg_bin, left, gigaam_model=model, segment_sec=segment_sec
            )
            client = transcribe_gigaam_channel(
                ffmpeg_bin, right, gigaam_model=model, segment_sec=segment_sec
            )
            elapsed = time.perf_counter() - started

        combined = f"MANAGER:\n{manager}\n\nCLIENT:\n{client}"
        metrics = text_metrics(combined)
        rows.append(
            {
                "file": str(file_path),
                "duration_sec": round(duration_sec, 3),
                "elapsed_sec": round(elapsed, 3),
                "rtf": round(elapsed / duration_sec, 4),
                **metrics,
            }
        )

    return {
        "model": f"gigaam:{model_name}",
        "model_load_sec": round(load_elapsed, 3),
        "rows": rows,
        "avg_elapsed_sec": round(mean(r["elapsed_sec"] for r in rows), 3),
        "avg_rtf": round(mean(r["rtf"] for r in rows), 4),
    }


def print_table(name: str, rows: List[Dict[str, Any]]) -> None:
    print(f"\n=== {name} ===")
    print("file | duration_s | elapsed_s | rtf | words | max_word_run")
    for row in rows:
        print(
            f"{Path(row['file']).name} | {row['duration_sec']:.1f} | {row['elapsed_sec']:.2f}"
            f" | {row['rtf']:.3f} | {row['words']} | {row['max_word_run']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark GigaAM vs MLX-Whisper on the same calls")
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument(
        "--whisper-model",
        default="mlx-community/whisper-large-v3-mlx",
    )
    parser.add_argument("--language", default="ru")
    parser.add_argument("--gigaam-model", default="v2_rnnt")
    parser.add_argument("--gigaam-segment-sec", type=int, default=20)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()

    files = [Path(p).resolve() for p in args.files]
    for path in files:
        if not path.exists():
            raise FileNotFoundError(path)

    whisper = benchmark_whisper(
        ffmpeg_bin=args.ffmpeg_bin,
        files=files,
        model=args.whisper_model,
        language=args.language,
    )
    gigaam = benchmark_gigaam(
        ffmpeg_bin=args.ffmpeg_bin,
        files=files,
        model_name=args.gigaam_model,
        segment_sec=args.gigaam_segment_sec,
    )

    result = {
        "files": [str(p) for p in files],
        "whisper": whisper,
        "gigaam": gigaam,
        "summary": {
            "avg_elapsed_ratio_gigaam_over_whisper": round(
                gigaam["avg_elapsed_sec"] / whisper["avg_elapsed_sec"], 4
            ),
            "avg_rtf_ratio_gigaam_over_whisper": round(
                gigaam["avg_rtf"] / whisper["avg_rtf"], 4
            ),
        },
    }

    out_path = Path(args.output_json).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print_table(whisper["model"], whisper["rows"])
    print_table(gigaam["model"], gigaam["rows"])
    print("\n=== summary ===")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
