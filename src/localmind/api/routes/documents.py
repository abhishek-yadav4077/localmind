"""Document chat endpoints (501 when the optional ``rag`` extra isn't installed)."""

from __future__ import annotations

from dataclasses import asdict
from typing import Annotated, Any

from fastapi import APIRouter, File, Response, UploadFile
from starlette.concurrency import run_in_threadpool

from localmind.api.deps import Ctx
from localmind.api.schemas import AskDocumentsRequest
from localmind.api.sse import sse_response

router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("")
async def list_documents(ctx: Ctx) -> list[dict[str, Any]]:
    rag = await run_in_threadpool(ctx.rag)
    return [asdict(d) for d in rag.store.list_documents()]


@router.post("", status_code=201)
async def upload_document(ctx: Ctx, file: Annotated[UploadFile, File()]) -> dict[str, Any]:
    rag = await run_in_threadpool(ctx.rag)
    data = await file.read()
    # Embedding is slow and blocking: keep it off the event loop.
    info = await run_in_threadpool(rag.store.add_document, file.filename or "upload.txt", data)
    return asdict(info)


@router.delete("/{doc_id}", status_code=204)
async def delete_document(doc_id: str, ctx: Ctx) -> Response:
    rag = await run_in_threadpool(ctx.rag)
    await run_in_threadpool(rag.store.delete_document, doc_id)
    return Response(status_code=204)


@router.post("/ask")
async def ask(body: AskDocumentsRequest, ctx: Ctx):
    """Events: ``sources``, ``token``*, ``done`` or ``error``."""
    rag = await run_in_threadpool(ctx.rag)
    return await sse_response(
        lambda: rag.stream(
            body.question,
            doc_ids=body.doc_ids,
            model=body.model,
            conversation_id=body.conversation_id,
            k=body.k,
        )
    )
