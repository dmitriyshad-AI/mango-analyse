from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from mango_mvp.amocrm_runtime.config import get_settings
from mango_mvp.amocrm_runtime.db import Base, engine
from mango_mvp.amocrm_runtime import models as _models  # noqa: F401
from mango_mvp.amocrm_runtime.routers.deals import router as deals_router
from mango_mvp.amocrm_runtime.routers.integrations import router as integrations_router
from mango_mvp.amocrm_runtime.routers.tallanto import router as tallanto_router

settings = get_settings()

if settings.agent_runtime_enabled:
    from mango_mvp.amocrm_runtime.routers.agent import router as agent_router


@asynccontextmanager
async def lifespan(_: FastAPI):
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(
    title="amoCRM Runtime",
    version="0.1.0",
    summary="Separate amoCRM runtime extracted from AI Office",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.api_cors_allow_origins),
    allow_credentials=bool(settings.api_cors_allow_origins),
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(integrations_router, prefix="/api")
app.include_router(deals_router, prefix="/api")
app.include_router(tallanto_router, prefix="/api")
if settings.agent_runtime_enabled:
    app.include_router(agent_router, prefix="/api")


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "amocrm-runtime",
        "status": "ok",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "service": "amocrm-runtime",
        "status": "ok",
    }
