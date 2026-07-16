from __future__ import annotations

import importlib
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
PARTS_MODULE = "mango_mvp.channels.subscription_llm_parts"


def test_subscription_llm_parts_exports_resolve_without_monolith() -> None:
    parts = importlib.import_module(PARTS_MODULE)
    facade = importlib.import_module("mango_mvp.channels.subscription_llm")

    assert set(parts.__all__) == set(facade.__all__)
    assert all(hasattr(parts, name) for name in parts.__all__)

    with pytest.raises(ModuleNotFoundError) as exc_info:
        importlib.import_module(f"{PARTS_MODULE}.monolith")
    assert exc_info.value.name == f"{PARTS_MODULE}.monolith"


def test_subscription_llm_parts_unknown_export_fails_loud(tmp_path: Path) -> None:
    copied_src = tmp_path / "src"
    shutil.copytree(ROOT / "src" / "mango_mvp", copied_src / "mango_mvp")

    facade_init = copied_src / "mango_mvp/channels/subscription_llm_parts/__init__.py"
    source = facade_init.read_text(encoding="utf-8")
    marker = "for _name in __all__:\n"
    assert source.count(marker) == 1
    facade_init.write_text(
        source.replace(
            marker,
            '__all__.append("__missing_export_probe__")\n\n' + marker,
        ),
        encoding="utf-8",
    )

    env = dict(os.environ)
    env["PYTHONPATH"] = str(copied_src)
    completed = subprocess.run(
        [sys.executable, "-c", f"import {PARTS_MODULE}"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "AttributeError" in completed.stderr
    assert "__missing_export_probe__" in completed.stderr
