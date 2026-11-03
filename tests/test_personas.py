import pytest

from localmind.chat import ChatService
from localmind.db import Database
from localmind.personas import DEFAULT_PERSONAS, PersonaService
from localmind.personas.service import PersonaNotFoundError


def test_seed_is_idempotent_and_keeps_user_edits(db: Database) -> None:
    svc = PersonaService(db)
    assert svc.seed_defaults() == len(DEFAULT_PERSONAS)
    tutor = next(p for p in svc.list() if p.name == "Tutor")
    svc.update(tutor.id, system_prompt="custom")
    assert svc.seed_defaults() == 0
    assert svc.get(tutor.id).system_prompt == "custom"


def test_crud(db: Database) -> None:
    svc = PersonaService(db)
    p = svc.create("Pirate", "Talk like a pirate.", "Arr")
    assert svc.get(p.id).name == "Pirate"
    svc.update(p.id, name="Captain")
    assert svc.get(p.id).name == "Captain"
    svc.delete(p.id)
    with pytest.raises(PersonaNotFoundError):
        svc.get(p.id)


def test_deleting_persona_detaches_conversations(db: Database, settings) -> None:
    personas = PersonaService(db)
    p = personas.create("Temp", "x")
    chat = ChatService(db, settings, llm_factory=lambda m, p: None)
    conv = chat.create_conversation(persona_id=p.id)
    personas.delete(p.id)
    assert chat.get_conversation(conv.id).persona_id is None
