from __future__ import annotations

from dataclasses import replace

import pytest

from mango_mvp.services.sync_amocrm import (
    LEGACY_AMOCRM_SYNC_DISABLED_MESSAGE,
    ensure_legacy_amocrm_sync_enabled,
)
from tests.test_dialogue_format import make_settings


def test_legacy_amocrm_sync_is_disabled_by_default_at_service_level() -> None:
    settings = make_settings()

    with pytest.raises(RuntimeError, match="Legacy amoCRM contact sync is disabled"):
        ensure_legacy_amocrm_sync_enabled(settings)


def test_legacy_amocrm_sync_allows_explicit_maintenance_opt_in() -> None:
    settings = replace(make_settings(), legacy_amocrm_sync_enabled=True)

    ensure_legacy_amocrm_sync_enabled(settings)


def test_legacy_amocrm_sync_disabled_message_points_to_current_runtime() -> None:
    assert "amocrm_runtime" in LEGACY_AMOCRM_SYNC_DISABLED_MESSAGE
    assert "LEGACY_AMOCRM_SYNC_ENABLED=true" in LEGACY_AMOCRM_SYNC_DISABLED_MESSAGE
