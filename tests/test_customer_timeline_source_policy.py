from __future__ import annotations

import pytest

from mango_mvp.customer_timeline.source_policy import (
    WAPPI_MAX_SOURCE_SYSTEM,
    WAPPI_TELEGRAM_SOURCE_SYSTEM,
    assert_bot_context_chunk_source_policy,
    is_bot_forbidden_source_system,
)


@pytest.mark.parametrize("source_system", [WAPPI_TELEGRAM_SOURCE_SYSTEM, WAPPI_MAX_SOURCE_SYSTEM])
def test_wappi_history_sources_are_manager_only(source_system: str) -> None:
    assert is_bot_forbidden_source_system(source_system) is True

    with pytest.raises(ValueError, match="allowed_for_bot=False"):
        assert_bot_context_chunk_source_policy(
            source_system=source_system,
            allowed_for_bot=True,
            requires_manager_review=False,
        )

    assert_bot_context_chunk_source_policy(
        source_system=source_system,
        allowed_for_bot=False,
        requires_manager_review=True,
    )
