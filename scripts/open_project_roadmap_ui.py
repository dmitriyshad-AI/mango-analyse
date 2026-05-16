#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import webbrowser


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    ui_path = project_root / "docs" / "project_roadmap_ui.html"
    if not ui_path.exists():
        print(f"Файл интерфейса не найден: {ui_path}")
        return 1

    webbrowser.open(ui_path.resolve().as_uri())
    print(f"Открыт интерфейс: {ui_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
