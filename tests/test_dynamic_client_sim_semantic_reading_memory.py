from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SIM = ROOT / "scripts" / "run_telegram_dynamic_client_sim.py"


def test_dynamic_client_sim_passes_semantic_reading_to_all_memory_updates() -> None:
    text = SIM.read_text(encoding="utf-8")

    assert "from mango_mvp.channels.subscription_llm_parts.semantic_reading import SemanticReading" in text
    assert "def _semantic_reading_memory_from_result" in text
    assert "def _semantic_reading_memory_from_turn" in text
    assert text.count("semantic_reading=_semantic_reading_memory_from_") == 3
    assert "semantic_reading=_semantic_reading_memory_from_result(result)" in text
    assert "semantic_reading=_semantic_reading_memory_from_result(framed)" in text
    assert "semantic_reading=_semantic_reading_memory_from_turn(turn)" in text
