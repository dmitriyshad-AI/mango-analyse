from mango_mvp.channels.subscription_llm_parts.codex_exec import build_codex_exec_command


def test_codex_exec_command_overrides_service_tier_default(monkeypatch, tmp_path):
    monkeypatch.delenv("MANGO_CODEX_SERVICE_TIER", raising=False)

    cmd = build_codex_exec_command(output_path=tmp_path / "out.json")

    assert 'service_tier="fast"' in cmd


def test_codex_exec_command_allows_service_tier_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("MANGO_CODEX_SERVICE_TIER", "flex")

    cmd = build_codex_exec_command(output_path=tmp_path / "out.json")

    assert 'service_tier="flex"' in cmd
