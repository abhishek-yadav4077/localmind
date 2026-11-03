"""FastAPI dependencies."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from localmind.bootstrap import AppContext


def get_ctx(request: Request) -> AppContext:
    """The app-wide context created at startup (see ``main.lifespan``)."""
    return request.app.state.ctx


Ctx = Annotated[AppContext, Depends(get_ctx)]
