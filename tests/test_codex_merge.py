from __future__ import annotations

import subprocess
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from mango_mvp.services.transcribe import TranscribeService
from tests.test_dialogue_format import make_settings


class CodexMergeTest(unittest.TestCase):
    def test_extract_json_payload_from_fenced_block(self) -> None:
        service = TranscribeService(make_settings())
        payload = service._extract_json_payload(
            "```json\n"
            '{"merged_text":"текст","selection":"MIX","confidence":0.8,"notes":"ok"}\n'
            "```"
        )
        self.assertEqual(payload.get("selection"), "MIX")
        self.assertEqual(payload.get("merged_text"), "текст")

    def test_merge_variant_pair_uses_codex_cli_provider(self) -> None:
        settings = replace(make_settings(), dual_merge_provider="codex_cli")
        service = TranscribeService(settings)
        with patch.object(
            service,
            "_merge_with_codex_cli",
            return_value={
                "text": "объединенный текст",
                "selection": "MIX",
                "confidence": 0.88,
                "notes": "ok",
                "provider": "codex_cli",
            },
        ):
            result = service._merge_variant_pair(
                "вариант один",
                "вариант два",
                speaker_label="Менеджер",
            )
        self.assertEqual(result.get("provider"), "codex_cli")
        self.assertEqual(result.get("selection"), "MIX")
        self.assertIn("similarity", result)

    def test_merge_variant_pair_codex_cli_fallback_to_rule(self) -> None:
        settings = replace(make_settings(), dual_merge_provider="codex_cli")
        service = TranscribeService(settings)
        with patch.object(service, "_merge_with_codex_cli", side_effect=RuntimeError("boom")):
            result = service._merge_variant_pair(
                "вариант один",
                "вариант два",
                speaker_label="Клиент",
            )
        self.assertEqual(result.get("provider"), "rule_fallback")
        self.assertIn("codex_cli_merge_failed", str(result.get("notes", "")))

    def test_merge_with_codex_cli_reads_output_last_message(self) -> None:
        settings = replace(
            make_settings(),
            codex_cli_command="codex",
            codex_merge_model="gpt-5-codex",
            codex_cli_timeout_sec=30,
        )
        service = TranscribeService(settings)

        def _fake_run(cmd: list[str], **kwargs):  # type: ignore[no-untyped-def]
            out_idx = cmd.index("--output-last-message") + 1
            out_path = Path(cmd[out_idx])
            out_path.write_text(
                '{"merged_text":"тест","selection":"A","confidence":0.91,"notes":"ok"}',
                encoding="utf-8",
            )
            self.assertIn("--model", cmd)
            self.assertIn("gpt-5-codex", cmd)
            return subprocess.CompletedProcess(cmd, 0, "", "")

        with patch("mango_mvp.services.transcribe.shutil.which", return_value="/usr/bin/codex"):
            with patch("mango_mvp.services.transcribe.subprocess.run", side_effect=_fake_run):
                merged = service._merge_with_codex_cli(
                    "вариант A",
                    "вариант B",
                    speaker_label="Менеджер",
                )
        self.assertEqual(merged.get("provider"), "codex_cli")
        self.assertEqual(merged.get("selection"), "A")
        self.assertEqual(merged.get("text"), "тест")


if __name__ == "__main__":
    unittest.main()
