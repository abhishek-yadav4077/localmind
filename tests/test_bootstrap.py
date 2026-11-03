from localmind.bootstrap import build_context
from localmind.config import Settings


def test_build_context_creates_tables_and_seeds_personas() -> None:
    ctx = build_context(Settings(_env_file=None, database_url="sqlite://"))
    assert {p.name for p in ctx.personas.list()} >= {"Tutor", "Code Reviewer"}
    assert ctx.tracing.enabled is False
    conv = ctx.chat.create_conversation()
    assert ctx.chat.get_conversation(conv.id).model == "llama3.2"
