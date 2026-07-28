from __future__ import annotations

import importlib


PINNED_BINDINGS = {
    "_optional_text": "mango_mvp.channels.subscription_llm_parts.provider",
    "_clamp_float": "mango_mvp.channels.output_verification_floor",
}


def test_subscription_llm_reexport_bindings_are_not_silently_reordered() -> None:
    package = importlib.import_module("mango_mvp.channels.subscription_llm_parts")
    violations: list[str] = []
    for name, expected_module in PINNED_BINDINGS.items():
        value = getattr(package, name, None)
        actual_module = getattr(value, "__module__", None)
        if actual_module != expected_module:
            violations.append(f"{name}: expected {expected_module}, got {actual_module}")
    assert violations == [], "Re-export binding changed:\n" + "\n".join(violations)
