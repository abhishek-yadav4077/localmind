"""Small HTTP client for the LocalMind API, including an SSE stream parser."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from typing import Any

import httpx

DEFAULT_API_URL = "http://localhost:8000"


class APIError(Exception):
    """An error response from the API (or the API being unreachable)."""

    def __init__(self, error: str, hint: str | None = None, status: int | None = None) -> None:
        super().__init__(error)
        self.error, self.hint, self.status = error, hint, status


def _raise_for(response: httpx.Response) -> None:
    if response.status_code < 400:
        return
    try:
        body = response.json()
        error = body.get("error") or body.get("detail") or response.text
        hint = body.get("hint")
    except ValueError:
        error, hint = response.text, None
    if isinstance(error, list):  # FastAPI validation errors
        error = "; ".join(f"{'.'.join(map(str, e.get('loc', [])))}: {e.get('msg')}" for e in error)
    raise APIError(str(error), hint, response.status_code)


class APIClient:
    """Typed-enough wrapper over the REST + SSE endpoints."""

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = (base_url or os.getenv("LOCALMIND_API_URL", DEFAULT_API_URL)).rstrip("/")
        self._http = httpx.Client(base_url=self.base_url, timeout=timeout, transport=transport)

    def _call(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            response = self._http.request(method, path, **kwargs)
        except httpx.ConnectError as exc:
            raise APIError(
                f"LocalMind API not reachable at {self.base_url}",
                hint="Start it with: uvicorn localmind.api.main:app --reload",
            ) from exc
        _raise_for(response)
        if response.status_code == 204 or not response.content:
            return None
        if response.headers.get("content-type", "").startswith("application/json"):
            return response.json()
        return response.text

    def get(self, path: str, **params: Any) -> Any:
        return self._call("GET", path, params={k: v for k, v in params.items() if v is not None})

    def post(self, path: str, json_body: Any = None, **kwargs: Any) -> Any:
        return self._call("POST", path, json=json_body, **kwargs)

    def patch(self, path: str, json_body: Any) -> Any:
        return self._call("PATCH", path, json=json_body)

    def delete(self, path: str) -> None:
        self._call("DELETE", path)

    def upload(self, path: str, filename: str, data: bytes, mime: str) -> Any:
        # Embedding a big PDF can take a while on CPU.
        return self._call("POST", path, files={"file": (filename, data, mime)}, timeout=600)

    def stream(self, path: str, json_body: Any) -> Iterator[tuple[str, dict[str, Any]]]:
        """POST and yield ``(event, data)`` pairs from a Server-Sent Events response."""
        # No read timeout: a cold model load can take a minute before the first token.
        timeout = httpx.Timeout(10.0, read=None)
        try:
            with self._http.stream("POST", path, json=json_body, timeout=timeout) as response:
                if response.status_code >= 400:
                    response.read()
                    _raise_for(response)
                yield from parse_sse(response.iter_lines())
        except httpx.ConnectError as exc:
            raise APIError(
                f"LocalMind API not reachable at {self.base_url}",
                hint="Start it with: uvicorn localmind.api.main:app --reload",
            ) from exc


def parse_sse(lines: Iterator[str]) -> Iterator[tuple[str, dict[str, Any]]]:
    """Minimal SSE parser: ``event:``/``data:`` fields, blank line ends an event."""
    event, data = "message", []
    for line in lines:
        if line.startswith(":"):  # comment / keep-alive ping
            continue
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].lstrip())
        elif not line.strip():
            if data:
                yield event, json.loads("\n".join(data))
            event, data = "message", []
    if data:
        yield event, json.loads("\n".join(data))
