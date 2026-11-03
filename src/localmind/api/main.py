"""FastAPI application factory.

Run locally:   uvicorn localmind.api.main:app --reload
Interactive docs at http://localhost:8000/docs
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sqlalchemy.exc import IntegrityError

from localmind import __version__
from localmind.api.routes import (
    agent,
    arena,
    chat,
    conversations,
    dashboard,
    documents,
    feedback,
    health,
    models,
    personas,
)
from localmind.bootstrap import AppContext, build_context
from localmind.config import get_logger
from localmind.exceptions import (
    FeatureUnavailableError,
    LocalMindError,
    ModelNotFoundError,
    ModelPullError,
    NotFoundError,
    OllamaUnavailableError,
)

logger = get_logger(__name__)

# Most specific first: the first matching class decides the status code.
_STATUS: list[tuple[type[LocalMindError], int]] = [
    (NotFoundError, 404),
    (ModelNotFoundError, 404),
    (OllamaUnavailableError, 503),
    (ModelPullError, 502),
    (FeatureUnavailableError, 501),
    (LocalMindError, 400),
]


def _error_body(message: str, hint: str | None = None) -> dict[str, str | None]:
    return {"error": message, "hint": hint}


def create_app(ctx: AppContext | None = None) -> FastAPI:
    """Build the app. Tests pass a prepared ``ctx``; production builds one at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.ctx = ctx or build_context()
        logger.info("api started", extra={"version": __version__})
        yield
        app.state.ctx.ollama.close()
        app.state.ctx.tracer.flush()

    app = FastAPI(
        title="LocalMind API",
        version=__version__,
        description="Private local LLM chat studio on Ollama, observed with LangSmith.",
        lifespan=lifespan,
    )

    @app.exception_handler(LocalMindError)
    async def handle_app_error(_: Request, exc: LocalMindError) -> JSONResponse:
        status = next(code for cls, code in _STATUS if isinstance(exc, cls))
        return JSONResponse(_error_body(exc.message, exc.hint), status_code=status)

    @app.exception_handler(IntegrityError)
    async def handle_integrity(_: Request, exc: IntegrityError) -> JSONResponse:
        # e.g. a persona name that already exists (UNIQUE constraint).
        return JSONResponse(_error_body("Conflicts with an existing record"), status_code=409)

    for module in (
        health,
        models,
        conversations,
        chat,
        arena,
        agent,
        documents,
        feedback,
        personas,
        dashboard,
    ):
        app.include_router(module.router)
    return app


app = create_app()
