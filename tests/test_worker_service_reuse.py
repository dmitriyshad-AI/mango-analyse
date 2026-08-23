from __future__ import annotations

import os
from unittest.mock import MagicMock, Mock, patch

from mango_mvp.services.worker import run_worker
from tests.test_dialogue_format import make_settings


def test_long_lived_gigaam_worker_reuses_one_transcribe_service() -> None:
    settings = make_settings()
    service = Mock()
    service.backfill_secondary_asr.side_effect = [
        {"processed": 1, "success": 1, "failed": 0},
        {"processed": 0, "success": 0, "failed": 0},
        {"processed": 0, "success": 0, "failed": 0},
    ]

    with (
        patch("mango_mvp.services.worker.TranscribeService", return_value=service) as factory,
        patch("mango_mvp.services.worker.controlled_worker_parent_lifeline"),
        patch("mango_mvp.services.worker.enforce_controlled_worker_stages"),
        patch("mango_mvp.services.worker.build_session_factory") as session_factory,
        patch("mango_mvp.services.worker.time.sleep"),
    ):
        session_factory.return_value.return_value.__enter__.return_value = Mock()
        result = run_worker(
            settings,
            stage_limit=10,
            once=False,
            stages=["backfill-second-asr"],
            poll_sec=1,
            max_idle_cycles=2,
        )

    assert result["cycles"] == 3
    assert service.backfill_secondary_asr.call_count == 3
    factory.assert_called_once_with(settings)


def test_batch_gigaam_model_is_loaded_before_first_database_session() -> None:
    settings = make_settings()
    service = Mock()
    events: list[str] = []
    service._get_gigaam_model.side_effect = lambda: events.append("model")
    service.backfill_secondary_asr.return_value = {
        "processed": 0,
        "success": 0,
        "failed": 0,
    }

    session_factory = Mock()
    session_factory.return_value = MagicMock()
    session_factory.return_value.__enter__.side_effect = lambda: events.append("database") or Mock()
    with (
        patch.dict(os.environ, {"GIGAAM_BATCH_SIZE": "4"}),
        patch("mango_mvp.services.worker.TranscribeService", return_value=service),
        patch("mango_mvp.services.worker.controlled_worker_parent_lifeline"),
        patch("mango_mvp.services.worker.enforce_controlled_worker_stages"),
        patch("mango_mvp.services.worker.build_session_factory", return_value=session_factory),
    ):
        run_worker(
            settings,
            stage_limit=1,
            once=True,
            stages=["backfill-second-asr"],
        )

    assert events[:2] == ["model", "database"]
