from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from localmind.memory import MemoryManager, MemoryPolicy, estimate_tokens
from tests.fakes import FakeOllamaChat


def turns(n: int, size: int = 400) -> list:
    msgs = []
    for i in range(n):
        msgs.append(HumanMessage(content=f"q{i} " + "x" * size))
        msgs.append(AIMessage(content=f"a{i} " + "y" * size))
    return msgs


def test_estimate_tokens() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("hi") == 1
    assert estimate_tokens("a" * 400) == 100


def test_budget_reserves_room_for_answer() -> None:
    policy = MemoryPolicy(reserve_output_tokens=512)
    assert policy.budget(4096, None) == 3584
    assert policy.budget(4096, 1000) == 3096


def test_no_summary_when_short_or_too_few_messages() -> None:
    mm = MemoryManager(FakeOllamaChat, MemoryPolicy(keep_recent=4))
    short = turns(2, size=10)
    assert not mm.needs_summary(short, len(short), num_ctx=4096)
    # Huge but only 2 messages: nothing old enough to fold.
    huge = turns(1, size=50_000)
    assert not mm.needs_summary(huge, len(huge), num_ctx=1024)


def test_summary_needed_near_limit() -> None:
    mm = MemoryManager(FakeOllamaChat, MemoryPolicy(keep_recent=4))
    history = turns(6, size=400)  # ~1200 tokens
    assert mm.needs_summary(history, len(history), num_ctx=1024)
    assert not mm.needs_summary(history, len(history), num_ctx=8192)


def test_split_keeps_recent_verbatim() -> None:
    mm = MemoryManager(FakeOllamaChat, MemoryPolicy(keep_recent=4))
    history = turns(5)
    old, recent = mm.split(history)
    assert len(old) == 6 and len(recent) == 4
    assert recent == history[-4:]


def test_summarize_merges_previous_summary() -> None:
    llm = FakeOllamaChat()
    mm = MemoryManager(lambda: llm)
    out = mm.summarize("User likes tea.", turns(1, size=5))
    assert out == "User is Ada, learning Python."
    sent = llm.calls[0]
    assert isinstance(sent[0], SystemMessage) and "long-term memory" in sent[0].content
    assert "User likes tea." in sent[1].content and "q0" in sent[1].content
