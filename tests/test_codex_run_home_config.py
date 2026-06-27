from pathlib import Path


def _read_flat_toml(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"')
    return values


def test_codex_run_home_config_defaults_to_flex() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / "configs" / "codex_run_home" / "config.toml"

    config = _read_flat_toml(config_path)

    assert config["service_tier"] == "flex"
    assert config["model"] == "gpt-5.5"
    assert config["model_reasoning_effort"] == "medium"
