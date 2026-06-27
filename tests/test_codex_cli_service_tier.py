from mango_mvp.utils.codex_cli import append_codex_service_tier, codex_service_tier


def test_codex_service_tier_defaults_to_flex(monkeypatch) -> None:
    monkeypatch.delenv("MANGO_CODEX_SERVICE_TIER", raising=False)

    assert codex_service_tier() == "flex"


def test_append_codex_service_tier_allows_fast_override(monkeypatch) -> None:
    monkeypatch.setenv("MANGO_CODEX_SERVICE_TIER", "fast")
    cmd = ["codex", "exec"]

    append_codex_service_tier(cmd)

    assert cmd[-2:] == ["-c", 'service_tier="fast"']
