from __future__ import annotations

import unittest

from mango_mvp.config import Settings
from mango_mvp.services.transcribe import TranscribeService


def make_settings(
    *,
    mono_mode: str = "off",
    openai_api_key: str | None = None,
) -> Settings:
    return Settings(
        database_url="sqlite:///test.db",
        openai_api_key=openai_api_key,
        transcribe_provider="mock",
        dual_transcribe_enabled=False,
        secondary_transcribe_provider=None,
        dual_merge_provider="rule",
        openai_merge_model="gpt-4o-mini",
        codex_merge_model="gpt-5-codex",
        codex_cli_command="codex",
        codex_cli_timeout_sec=120,
        codex_reasoning_effort="medium",
        dual_merge_similarity_threshold=0.985,
        analyze_provider="mock",
        openai_transcribe_model="gpt-4o-transcribe",
        mlx_whisper_model="mlx-community/whisper-large-v3-mlx",
        mlx_condition_on_previous_text=False,
        mlx_word_timestamps=True,
        gigaam_model="v2_rnnt",
        gigaam_device="cpu",
        gigaam_segment_sec=20,
        openai_analysis_model="gpt-4o-mini",
        analyze_ollama_num_predict=500,
        ollama_base_url="http://127.0.0.1:11434",
        ollama_model="gpt-oss:20b",
        ollama_think="medium",
        ollama_temperature=0.0,
        transcribe_language="ru",
        transcript_export_dir="transcripts",
        split_stereo_channels=True,
        stereo_overlap_similarity_threshold=0.97,
        stereo_overlap_min_chars=80,
        mono_role_assignment_mode=mono_mode,
        mono_role_assignment_min_confidence=0.62,
        mono_role_assignment_llm_threshold=0.72,
        openai_role_assign_model="gpt-4o-mini",
        max_workers=2,
        transcribe_max_attempts=3,
        resolve_max_attempts=2,
        analyze_max_attempts=3,
        sync_max_attempts=3,
        resolve_min_duration_sec=30,
        resolve_llm_trigger_score=75,
        resolve_accept_score=75,
        resolve_llm_provider="ollama",
        resolve_dialogue_mode="dialogue",
        resolve_llm_for_risky=False,
        resolve_rescue_provider=None,
        resolve_rescue_dual_enabled=False,
        resolve_postfilter_same_ts=True,
        resolve_risky_same_ts_threshold=2,
        resolve_aggressive_rescue_for_risky=True,
        retry_base_delay_sec=30,
        worker_poll_sec=10,
        worker_max_idle_cycles=30,
        amocrm_base_url=None,
        amocrm_access_token=None,
        amocrm_refresh_token=None,
        amocrm_client_id=None,
        amocrm_client_secret=None,
        amocrm_redirect_uri=None,
        amocrm_token_cache_path=".amocrm_tokens.json",
        amocrm_interests_field_id=None,
        amocrm_student_grade_field_id=None,
        amocrm_target_product_field_id=None,
        amocrm_personal_offer_field_id=None,
        amocrm_budget_field_id=None,
        amocrm_timeline_field_id=None,
        amocrm_next_step_field_id=None,
        amocrm_followup_score_field_id=None,
        amocrm_task_type_id=None,
        amocrm_task_responsible_user_id=None,
        sync_dry_run=True,
        follow_up_task_threshold=70,
    )


class DialogueFormatTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = TranscribeService(make_settings())

    def test_stereo_segments_include_exact_timecodes(self) -> None:
        manager_segments = [
            {"start": 0.2, "text": "Здравствуйте"},
            {"start": 4.4, "text": "Давайте проверим"},
        ]
        client_segments = [{"start": 2.1, "text": "Да, слушаю"}]

        lines = self.service._build_dialogue_lines(
            "Иван",
            manager_segments,
            client_segments,
            manager_fallback_text="",
            client_fallback_text="",
            call_duration_sec=20.0,
        )

        self.assertEqual(len(lines), 3)
        self.assertTrue(lines[0].startswith("[00:00.2] Менеджер (Иван): Здравствуйте"))
        self.assertTrue(lines[1].startswith("[00:02.1] Клиент: Да, слушаю"))
        self.assertTrue(lines[2].startswith("[00:04.4] Менеджер (Иван): Давайте проверим"))
        self.assertTrue(all("[~" not in line for line in lines))

    def test_stereo_fallback_uses_estimated_timecodes(self) -> None:
        lines = self.service._build_dialogue_lines(
            "Иван",
            manager_segments=None,
            client_segments=None,
            manager_fallback_text="Добрый день. Расскажите, пожалуйста.",
            client_fallback_text="Да, слушаю.",
            call_duration_sec=30.0,
        )

        self.assertGreaterEqual(len(lines), 3)
        self.assertTrue(all(line.startswith("[~") for line in lines))
        self.assertTrue(any("Менеджер (Иван):" in line for line in lines))
        self.assertTrue(any("Клиент:" in line for line in lines))

    def test_mono_fallback_marks_unknown_speaker(self) -> None:
        turns = self.service._build_mono_turns(
            full_segments=None,
            full_fallback_text="Алло. Добрый день.",
            call_duration_sec=12.0,
        )
        lines = self.service._build_mono_dialogue_lines_from_turns(
            turns,
            "Спикер (не определен)",
        )

        self.assertEqual(len(lines), 2)
        self.assertTrue(all(line.startswith("[~") for line in lines))
        self.assertTrue(all("Спикер (не определен):" in line for line in lines))

    def test_stereo_similarity_guard(self) -> None:
        mirrored = (
            "Алло добрый день это тестовая фраза которая повторяется один в один "
            "и содержит достаточно символов чтобы сработал фильтр похожести каналов."
        )
        should_fallback, similarity = self.service._should_fallback_to_mono_from_stereo(
            mirrored,
            mirrored,
        )

        self.assertTrue(should_fallback)
        self.assertAlmostEqual(similarity, 1.0, places=6)

    def test_stereo_identical_short_guard(self) -> None:
        mirrored = "Продолжаем дозваниваться. Оставайтесь на линии."
        should_fallback, similarity = self.service._should_fallback_to_mono_from_stereo(
            mirrored,
            mirrored,
        )

        self.assertTrue(should_fallback)
        self.assertAlmostEqual(similarity, 1.0, places=6)

    def test_stereo_crosstalk_dedupe_removes_mirrored_lines(self) -> None:
        lines = [
            "[00:00.0] Менеджер (Иван): Добрый день, это учебный центр.",
            "[00:00.2] Клиент: Добрый день, это учебный центр.",
            "[00:03.0] Клиент: Меня интересует курс по математике.",
            "[00:03.3] Менеджер (Иван): Меня интересует курс по математике.",
            "[00:06.0] Менеджер (Иван): Подскажите, какой класс?",
            "[00:07.0] Клиент: Девятый класс.",
        ]

        dedupe = self.service._dedupe_stereo_cross_talk(lines)

        self.assertEqual(int(dedupe["dropped"]), 2)
        cleaned_lines = dedupe["dialogue_lines"]
        self.assertEqual(len(cleaned_lines), 4)
        joined = "\n".join(cleaned_lines)
        self.assertEqual(joined.count("Добрый день, это учебный центр."), 1)
        self.assertEqual(joined.count("Меня интересует курс по математике."), 1)

    def test_stereo_sequence_fix_swaps_answer_before_question(self) -> None:
        lines = [
            "[00:41.5] Клиент: Десятый класс.",
            "[00:41.5] Менеджер (Иван): Подскажите, какой класс вас интересует?",
            "[00:50.0] Менеджер (Иван): Отлично, спасибо.",
        ]

        fixed = self.service._resequence_dialogue_lines(lines)

        self.assertEqual(int(fixed["swapped"]), 1)
        fixed_lines = fixed["dialogue_lines"]
        self.assertIn("Подскажите, какой класс вас интересует?", fixed_lines[0])
        self.assertIn("Десятый класс.", fixed_lines[1])

    def test_artifact_only_lines_are_removed(self) -> None:
        lines = [
            "[00:10.0] Менеджер (Иван): Продолжение следует...",
            "[00:11.5] Клиент: Хорошо, спасибо.",
        ]

        cleaned = self.service._drop_artifact_only_lines(lines)

        self.assertEqual(int(cleaned["dropped"]), 1)
        out = cleaned["dialogue_lines"]
        self.assertEqual(len(out), 1)
        self.assertIn("Клиент: Хорошо, спасибо.", out[0])

    def test_adjacent_cross_speaker_echo_is_dropped(self) -> None:
        lines = [
            "[00:20.0] Менеджер (Иван): Я могу получить какие-то результаты за первое полугодие?",
            "[00:20.0] Клиент: Я могу получить какие-то результаты за первое полугодие?",
            "[00:22.0] Клиент: Да, конечно.",
        ]

        deduped = self.service._dedupe_adjacent_cross_speaker_echo(lines)

        self.assertEqual(int(deduped["dropped"]), 1)
        out = deduped["dialogue_lines"]
        self.assertEqual(len(out), 2)
        joined = "\n".join(out)
        self.assertEqual(
            joined.count("Я могу получить какие-то результаты за первое полугодие?"),
            1,
        )

    def test_segments_to_timeline_uses_word_timestamps(self) -> None:
        raw_segments = [
            {
                "start": 0.0,
                "text": "черновик",
                "words": [
                    {"start": 0.2, "end": 0.4, "word": "Здравствуйте"},
                    {"start": 0.5, "end": 0.7, "word": "как"},
                    {"start": 0.8, "end": 1.0, "word": "дела?"},
                    {"start": 2.1, "end": 2.4, "word": "Подскажите"},
                    {"start": 2.5, "end": 2.7, "word": "класс."},
                ],
            }
        ]

        timeline = self.service._segments_to_timeline(raw_segments, "Клиент")

        self.assertEqual(len(timeline), 2)
        self.assertAlmostEqual(float(timeline[0][0]), 0.2, places=2)
        self.assertIn("Здравствуйте", timeline[0][3])
        self.assertAlmostEqual(float(timeline[1][0]), 2.1, places=2)
        self.assertIn("Подскажите", timeline[1][3])

    def test_rule_based_mono_role_assignment(self) -> None:
        service = TranscribeService(make_settings(mono_mode="rule"))
        turns = [
            {"start": 0.0, "approximate": False, "text": "Добрый день, вас беспокоит учебный центр."},
            {"start": 2.3, "approximate": False, "text": "Здравствуйте, а у вас есть курс по математике?"},
            {"start": 5.0, "approximate": False, "text": "Да, подскажите пожалуйста ваш класс."},
            {"start": 8.1, "approximate": False, "text": "10 класс, можно информацию на почту?"},
        ]
        warnings: list[str] = []
        assigned = service._assign_roles_for_mono(turns, "Иванов", warnings)

        self.assertIsNotNone(assigned)
        assert assigned is not None
        self.assertIn("Менеджер (Иванов):", "\n".join(assigned["dialogue_lines"]))
        self.assertIn("Клиент:", "\n".join(assigned["dialogue_lines"]))
        self.assertGreaterEqual(float(assigned["meta"]["confidence"]), 0.62)

    def test_openai_selective_without_key_uses_rule_fallback(self) -> None:
        service = TranscribeService(make_settings(mono_mode="openai_selective", openai_api_key=None))
        turns = [
            {"start": 0.0, "approximate": False, "text": "Добрый день, вас беспокоит учебный центр."},
            {"start": 2.0, "approximate": False, "text": "Здравствуйте, можно стоимость курса?"},
        ]
        warnings: list[str] = []
        assigned = service._assign_roles_for_mono(turns, "Петров", warnings)

        self.assertIsNotNone(assigned)
        assert assigned is not None
        provider = str(assigned["meta"]["provider"])
        self.assertIn(provider, {"rule_high_conf", "rule_fallback"})
        if provider == "rule_fallback":
            self.assertTrue(
                any("OPENAI_API_KEY missing" in msg for msg in warnings),
                msg=f"warnings={warnings}",
            )


if __name__ == "__main__":
    unittest.main()
