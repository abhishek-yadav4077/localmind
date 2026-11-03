"""Persona CRUD and seeding."""

from __future__ import annotations

from collections.abc import Sequence

from localmind.config import get_logger
from localmind.db import Database, Persona
from localmind.db import repositories as repo
from localmind.exceptions import NotFoundError
from localmind.personas.defaults import DEFAULT_PERSONAS

logger = get_logger(__name__)


class PersonaNotFoundError(NotFoundError):
    """No persona with that id."""


class PersonaService:
    """Create, read, update and delete personas."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def seed_defaults(self) -> int:
        """Insert built-in personas that don't exist yet. Returns how many were added.

        Matching is by name, so user edits to a built-in persona survive restarts.
        """
        added = 0
        with self.db.session() as s:
            for spec in DEFAULT_PERSONAS:
                if repo.get_persona_by_name(s, spec["name"]) is None:
                    s.add(Persona(**spec, is_builtin=True))
                    added += 1
        if added:
            logger.info("seeded personas", extra={"count": added})
        return added

    def list(self) -> Sequence[Persona]:
        with self.db.session() as s:
            return repo.list_personas(s)

    def get(self, persona_id: int) -> Persona:
        with self.db.session() as s:
            persona = repo.get_persona(s, persona_id)
            if persona is None:
                raise PersonaNotFoundError(f"Persona {persona_id} not found")
            return persona

    def create(self, name: str, system_prompt: str, description: str = "") -> Persona:
        with self.db.session() as s:
            persona = Persona(name=name, system_prompt=system_prompt, description=description)
            s.add(persona)
            s.flush()
            return persona

    def update(
        self,
        persona_id: int,
        *,
        name: str | None = None,
        system_prompt: str | None = None,
        description: str | None = None,
    ) -> Persona:
        with self.db.session() as s:
            persona = repo.get_persona(s, persona_id)
            if persona is None:
                raise PersonaNotFoundError(f"Persona {persona_id} not found")
            if name is not None:
                persona.name = name
            if system_prompt is not None:
                persona.system_prompt = system_prompt
            if description is not None:
                persona.description = description
            s.flush()
            return persona

    def delete(self, persona_id: int) -> None:
        """Delete a persona; conversations using it fall back to no persona (SET NULL)."""
        with self.db.session() as s:
            persona = repo.get_persona(s, persona_id)
            if persona is None:
                raise PersonaNotFoundError(f"Persona {persona_id} not found")
            s.delete(persona)
