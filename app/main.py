from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api import routes_sessions, routes_chat, routes_traces
from app.api.errors import HelixError, helix_error_handler
from app.db.session import init_db
from app.obs.logging import configure_logging


import os
from app.settings import settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    
    # Ensure data directories exist (for Render/Docker)
    if "sqlite" in settings.database_url:
        db_path = settings.database_url.split("///")[-1]
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    os.makedirs(os.path.abspath(settings.chroma_persist_dir), exist_ok=True)

    # Ensure ADK can find the API key
    if settings.google_api_key:
        os.environ["GOOGLE_API_KEY"] = settings.google_api_key
    await init_db()
    yield


app = FastAPI(title="Helix SROP", version="0.1.0", lifespan=lifespan)

app.add_exception_handler(HelixError, helix_error_handler)  # type: ignore[arg-type]

app.include_router(routes_sessions.router, prefix="/v1")
app.include_router(routes_chat.router, prefix="/v1")
app.include_router(routes_traces.router, prefix="/v1")


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}
