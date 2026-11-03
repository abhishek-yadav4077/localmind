"""Server-Sent Events helpers.

SSE (one-way, text/event-stream) fits token streaming better than WebSockets:
plain HTTP, works through proxies, trivial to consume, and the client never
needs to send anything mid-stream.

Wire format: ``event: <type>`` + ``data: <json>`` per event, where ``type`` is
the event model's ``type`` field (token, done, error, ...).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from starlette.concurrency import run_in_threadpool

from localmind.config import get_logger

logger = get_logger(__name__)
_END = object()


def _format(event: BaseModel | dict[str, Any]) -> dict[str, str]:
    if isinstance(event, BaseModel):
        return {"event": getattr(event, "type", "message"), "data": event.model_dump_json()}
    return {"event": str(event.get("type", "message")), "data": _json(event)}


def _json(obj: Any) -> str:
    import json

    return json.dumps(obj, default=str)


async def sse_response(make_events: Callable[[], Iterator[Any]]) -> EventSourceResponse:
    """Stream a service generator as SSE.

    The first event is computed *before* the response starts. Service
    generators validate input on their first step, so a bad request becomes a
    proper HTTP 4xx (via the app's exception handlers) instead of a 200 stream
    that dies immediately.
    """
    events = make_events()
    first = await run_in_threadpool(next, events, _END)

    def stream() -> Iterator[dict[str, str]]:
        if first is _END:
            return
        yield _format(first)
        try:
            for event in events:
                yield _format(event)
        except Exception as exc:
            logger.exception("stream failed mid-way")
            yield _format({"type": "error", "error": f"{type(exc).__name__}: {exc}"})

    # ping keeps idle proxies from closing the connection while a model loads.
    return EventSourceResponse(stream(), ping=15)
