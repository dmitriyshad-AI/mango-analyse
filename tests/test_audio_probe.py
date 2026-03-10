from __future__ import annotations

import json
import tempfile
import unittest
import wave
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

from mango_mvp.utils.audio import probe_audio


class AudioProbeTest(unittest.TestCase):
    def test_probe_uses_ffprobe_when_available(self) -> None:
        payload = {
            "streams": [{"codec_name": "mp3", "sample_rate": "8000", "channels": 2}],
            "format": {"duration": "12.34"},
        }
        with patch("mango_mvp.utils.audio.shutil.which", return_value="/usr/bin/ffprobe"):
            with patch(
                "mango_mvp.utils.audio.subprocess.run",
                return_value=CompletedProcess(args=[], returncode=0, stdout=json.dumps(payload), stderr=""),
            ):
                meta = probe_audio(Path("/tmp/a.mp3"))
        self.assertEqual(meta["codec_name"], "mp3")
        self.assertEqual(meta["sample_rate"], 8000)
        self.assertEqual(meta["channels"], 2)
        self.assertAlmostEqual(float(meta["duration_sec"] or 0), 12.34, places=2)

    def test_probe_falls_back_to_afinfo(self) -> None:
        afinfo_out = (
            "File type ID:   MPG3\n"
            "Data format:     2 ch,   8000 Hz, .mp3 (0x00000000)\n"
            "estimated duration: 56.520000 sec\n"
        )

        def fake_which(name: str) -> str | None:
            if name == "ffprobe":
                return None
            if name == "afinfo":
                return "/usr/bin/afinfo"
            return None

        with patch("mango_mvp.utils.audio.shutil.which", side_effect=fake_which):
            with patch(
                "mango_mvp.utils.audio.subprocess.run",
                return_value=CompletedProcess(args=[], returncode=0, stdout=afinfo_out, stderr=""),
            ):
                meta = probe_audio(Path("/tmp/a.mp3"))

        self.assertEqual(meta["codec_name"], "mp3")
        self.assertEqual(meta["sample_rate"], 8000)
        self.assertEqual(meta["channels"], 2)
        self.assertAlmostEqual(float(meta["duration_sec"] or 0), 56.52, places=2)

    def test_probe_falls_back_to_wave_for_wav(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_probe_wav_") as td:
            wav_path = Path(td) / "sample.wav"
            with wave.open(str(wav_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(8000)
                wf.writeframes(b"\x00\x00" * 8000)

            with patch("mango_mvp.utils.audio.shutil.which", return_value=None):
                meta = probe_audio(wav_path)

        self.assertEqual(meta["codec_name"], "pcm_s16le")
        self.assertEqual(meta["sample_rate"], 8000)
        self.assertEqual(meta["channels"], 1)
        self.assertAlmostEqual(float(meta["duration_sec"] or 0), 1.0, places=2)


if __name__ == "__main__":
    unittest.main()
