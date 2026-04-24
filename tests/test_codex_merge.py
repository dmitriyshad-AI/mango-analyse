from __future__ import annotations

import subprocess
import tempfile
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
            codex_transcribe_model="gpt-5-codex",
            codex_cli_timeout_sec=30,
            llm_cache_enabled=False,
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
            return subprocess.CompletedProcess(cmd, 0, "", "tokens used\n710\n")

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
        self.assertEqual(merged.get("tokens_used_actual"), 710)
        duration_sec = merged.get("duration_sec")
        self.assertIsInstance(duration_sec, float)
        self.assertGreaterEqual(duration_sec, 0.0)

    def test_parse_codex_tokens_used(self) -> None:
        service = TranscribeService(make_settings())
        self.assertEqual(service._parse_codex_tokens_used("tokens used\n22 301\n"), 22301)
        self.assertEqual(service._parse_codex_tokens_used("tokens used\n22\xa0301\n"), 22301)
        self.assertIsNone(service._parse_codex_tokens_used("no tokens here"))

    def test_merge_with_codex_cli_uses_response_cache_on_repeat(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_transcribe_cache_") as td:
            settings = replace(
                make_settings(),
                codex_cli_command="codex",
                codex_transcribe_model="gpt-5-codex",
                codex_cli_timeout_sec=30,
                llm_cache_enabled=True,
                llm_cache_dir=str(Path(td) / "llm-cache"),
            )
            service = TranscribeService(settings)
            state = {"calls": 0}

            def _fake_run(cmd: list[str], **kwargs):  # type: ignore[no-untyped-def]
                state["calls"] += 1
                out_idx = cmd.index("--output-last-message") + 1
                out_path = Path(cmd[out_idx])
                out_path.write_text(
                    '{"merged_text":"тест","selection":"A","confidence":0.91,"notes":"ok"}',
                    encoding="utf-8",
                )
                return subprocess.CompletedProcess(cmd, 0, "", "")

            with patch("mango_mvp.services.transcribe.shutil.which", return_value="/usr/bin/codex"):
                with patch("mango_mvp.services.transcribe.subprocess.run", side_effect=_fake_run):
                    first = service._merge_with_codex_cli(
                        "вариант A",
                        "вариант B",
                        speaker_label="Менеджер",
                    )
                    second = service._merge_with_codex_cli(
                        "вариант A",
                        "вариант B",
                        speaker_label="Менеджер",
                    )

        self.assertEqual(state["calls"], 1)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
