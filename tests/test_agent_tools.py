import pytest

from localmind.agent.tools import (
    build_tools,
    calculator,
    current_datetime,
    make_search_past_conversations,
    safe_eval,
)
from localmind.db import Database
from localmind.db import repositories as repo


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ("2 + 3 * 4", 14),
        ("2^10", 1024),
        ("(1 + 2) ** 2 / 3", 3.0),
        ("sqrt(16) + abs(-2)", 6.0),
        ("round(2.567, 2)", 2.57),
        ("1,000 * 3", 3000),
        ("-pi", -3.141592653589793),
        ("17 % 5", 2),
    ],
)
def test_safe_eval(expr: str, expected: float) -> None:
    assert safe_eval(expr) == pytest.approx(expected)


@pytest.mark.parametrize(
    "expr",
    ["__import__('os').system('ls')", "().__class__", "open('x')", "9**9**9", "a + 1", "lambda: 1"],
)
def test_safe_eval_rejects_unsafe_or_huge(expr: str) -> None:
    with pytest.raises((ValueError, SyntaxError)):
        safe_eval(expr)


def test_calculator_tool_formats_and_reports_errors() -> None:
    assert calculator.invoke({"expression": "6*7"}) == "6*7 = 42"
    assert calculator.invoke({"expression": "1/0"}).startswith("Error")


def test_current_datetime() -> None:
    out = current_datetime.invoke({"timezone": "UTC"})
    assert "date: " in out and "weekday: " in out and "UTC" in out
    assert current_datetime.invoke({"timezone": "Mars/Base"}).startswith("Error")


def test_search_past_conversations_excludes_current(db: Database) -> None:
    with db.session() as s:
        old = repo.create_conversation(s, model="m", title="Docker notes")
        repo.add_message(s, old.id, "user", "How do Docker volumes persist data?")
        current = repo.create_conversation(s, model="m")
        repo.add_message(s, current.id, "user", "remind me about docker")
        current_id = current.id
    tool = make_search_past_conversations(db, exclude_conversation_id=current_id)
    out = tool.invoke({"query": "docker"})
    assert "Docker notes" in out and "volumes" in out
    assert "remind me" not in out
    assert "No past messages" in tool.invoke({"query": "kubernetes"})


def test_web_search_is_opt_in(db: Database) -> None:
    assert "web_search" not in [t.name for t in build_tools(db)]
    assert "web_search" in [t.name for t in build_tools(db, web_search_enabled=True)]
