from __future__ import annotations

import uvicorn

from mango_mvp.amocrm_runtime.config import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "mango_mvp.amocrm_runtime.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )


if __name__ == "__main__":
    main()
