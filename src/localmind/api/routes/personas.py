"""Persona CRUD."""

from __future__ import annotations

from fastapi import APIRouter, Response

from localmind.api.deps import Ctx
from localmind.api.schemas import PersonaIn, PersonaOut, PersonaPatch

router = APIRouter(prefix="/personas", tags=["personas"])


@router.get("")
def list_personas(ctx: Ctx) -> list[PersonaOut]:
    return [PersonaOut.model_validate(p) for p in ctx.personas.list()]


@router.post("", status_code=201)
def create_persona(body: PersonaIn, ctx: Ctx) -> PersonaOut:
    return PersonaOut.model_validate(
        ctx.personas.create(body.name, body.system_prompt, body.description)
    )


@router.patch("/{persona_id}")
def update_persona(persona_id: int, body: PersonaPatch, ctx: Ctx) -> PersonaOut:
    return PersonaOut.model_validate(
        ctx.personas.update(persona_id, **body.model_dump(exclude_unset=True))
    )


@router.delete("/{persona_id}", status_code=204)
def delete_persona(persona_id: int, ctx: Ctx) -> Response:
    ctx.personas.delete(persona_id)
    return Response(status_code=204)
