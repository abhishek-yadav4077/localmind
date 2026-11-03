from localmind.arena import ArenaService
from localmind.chat import ChatService
from localmind.db.analytics import dashboard_stats
from localmind.tracing import disabled_tracer
from localmind.tracing.feedback import FeedbackService
from tests.fakes import FakeFactory


def test_dashboard_empty(db) -> None:
    stats = dashboard_stats(db)
    assert stats["totals"]["messages"] == 0
    assert stats["feedback"]["positive_ratio"] is None
    assert stats["per_model"] == [] and stats["daily"] == []


def test_dashboard_aggregates(db, settings) -> None:
    factory = FakeFactory()
    chat = ChatService(db, settings, llm_factory=factory)
    fb = FeedbackService(db, disabled_tracer(settings))
    a = chat.create_conversation(model="llama3.2")
    m1 = chat.reply(a.id, "one")
    m2 = chat.reply(a.id, "two")
    b = chat.create_conversation(model="mistral")
    m3 = chat.reply(b.id, "three")
    fb.submit(m1.id, True)
    fb.submit(m2.id, True)
    fb.submit(m3.id, False)
    ArenaService(db, settings, llm_factory=factory).run_battle("q", ["x", "y"])

    stats = dashboard_stats(db)
    assert stats["totals"]["conversations"] == 2
    assert stats["totals"]["assistant_messages"] == 3
    assert stats["totals"]["arena_battles"] == 1 and stats["totals"]["arena_votes"] == 0
    rows = {r["model"]: r for r in stats["per_model"]}
    assert rows["llama3.2"]["messages"] == 2 and rows["llama3.2"]["thumbs_up"] == 2
    assert rows["mistral"]["thumbs_down"] == 1
    assert rows["llama3.2"]["avg_tokens_per_sec"] is not None
    assert stats["feedback"] == {"thumbs_up": 2, "thumbs_down": 1, "positive_ratio": 0.667}
    assert stats["per_mode"] == {"chat": 3, "arena": 2}
    assert sum(d["messages"] for d in stats["daily"]) == 3
