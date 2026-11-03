"""Agent tools.

Every tool returns a *string*: that is what the model reads back, so the
format is designed for a small model to parse (short, labelled lines).

Only ``web_search`` touches the network. It is excluded unless the user turns
it on explicitly, to keep the app fully local by default.
"""

from __future__ import annotations

import ast
import math
import operator
import re
from collections.abc import Callable
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain_core.tools import BaseTool, tool

from localmind.db import Database
from localmind.db import repositories as repo

# ---------------------------------------------------------------- calculator
# A whitelist AST evaluator: ``eval`` on model output would be remote code
# execution, since the "input" is generated text we don't control.
_BIN_OPS: dict[type, Callable] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type, Callable] = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS: dict[str, Callable] = {
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "abs": abs,
    "round": round,
    "floor": math.floor,
    "ceil": math.ceil,
}
_CONSTS = {"pi": math.pi, "e": math.e}
MAX_EXPONENT = 1000  # stops "9**9**9" from hanging the process


def safe_eval(expression: str) -> float | int:
    """Evaluate an arithmetic expression using only whitelisted operations."""

    def ev(node: ast.AST) -> float | int:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
                raise ValueError("exponent too large")
            return _BIN_OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](ev(node.operand))
        if isinstance(node, ast.Name) and node.id in _CONSTS:
            return _CONSTS[node.id]
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _FUNCS
            and not node.keywords
        ):
            return _FUNCS[node.func.id](*(ev(a) for a in node.args))
        raise ValueError(f"unsupported expression element: {ast.dump(node)[:40]}")

    # Models often write "^" for power, "×" for multiply and "1,000" for 1000
    # (only strip commas that are thousands separators, not argument separators).
    cleaned = re.sub(r"(?<=\d),(?=\d{3}\b)", "", expression.replace("^", "**").replace("×", "*"))
    return ev(ast.parse(cleaned.strip(), mode="eval"))


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression exactly. Use this for ANY math instead of
    computing in your head. Supports + - * / // % ** parentheses, sqrt, log, log10,
    exp, sin, cos, tan, abs, round, floor, ceil, pi, e. Example: "sqrt(2) * (3 + 4)**2".
    """
    try:
        result = safe_eval(expression)
    except (ValueError, SyntaxError, TypeError, ZeroDivisionError, OverflowError) as exc:
        return f"Error: could not evaluate {expression!r}: {exc}"
    if isinstance(result, float) and result.is_integer() and abs(result) < 1e15:
        result = int(result)
    return f"{expression} = {result}"


# -------------------------------------------------------------- date / time
@tool
def current_datetime(timezone: str = "") -> str:
    """Get the current date, time and weekday. Use this whenever the answer depends on
    today's date. Optionally pass an IANA timezone like "Asia/Kolkata" or "UTC";
    leave empty for the computer's local time.
    """
    try:
        tz = ZoneInfo(timezone) if timezone else None
    except ZoneInfoNotFoundError:
        return f"Error: unknown timezone {timezone!r}. Use an IANA name like 'Europe/London'."
    now = datetime.now(tz).astimezone(tz)
    return (
        f"date: {now:%Y-%m-%d}\nweekday: {now:%A}\ntime: {now:%H:%M:%S}\n"
        f"timezone: {now.tzname()} (UTC{now:%z})"
    )


# ------------------------------------------------------- past conversations
def make_search_past_conversations(
    db: Database, exclude_conversation_id: str | None = None
) -> BaseTool:
    """Build the history-search tool bound to a database (closure over ``db``)."""

    @tool
    def search_past_conversations(query: str) -> str:
        """Search the user's previous chats with this app by keyword. Use this when
        the user refers to something discussed before ("what did we say about X",
        "my earlier question"). Use short keyword queries, e.g. "docker volumes".
        """
        with db.session() as s:
            hits = [
                (m, m.conversation.title)
                for m in repo.search_messages(s, query, limit=12)
                if m.conversation_id != exclude_conversation_id
            ][:6]
            if not hits:
                return f"No past messages matched {query!r}."
            lines = []
            for m, title in hits:
                snippet = " ".join(m.content.split())[:240]
                lines.append(f"- [{m.created_at:%Y-%m-%d}] chat '{title}' ({m.role}): {snippet}")
        return "Matches from past conversations:\n" + "\n".join(lines)

    return search_past_conversations


# --------------------------------------------------------------- web search
@tool
def web_search(query: str) -> str:
    """Search the web (DuckDuckGo) for recent or factual information you don't know.
    Returns titles, URLs and snippets. Cite the URLs you use.
    """
    try:
        from ddgs import DDGS  # imported lazily: only needed when web search is enabled

        results = DDGS().text(query, max_results=5)
    except Exception as exc:
        return f"Error: web search failed ({type(exc).__name__}: {exc})"
    if not results:
        return f"No web results for {query!r}."
    return "\n".join(
        f"- {r.get('title', '')}\n  url: {r.get('href', '')}\n  {r.get('body', '')[:300]}"
        for r in results
    )


def build_tools(
    db: Database, *, web_search_enabled: bool = False, exclude_conversation_id: str | None = None
) -> list[BaseTool]:
    """The tool set for one agent run."""
    tools: list[BaseTool] = [
        calculator,
        current_datetime,
        make_search_past_conversations(db, exclude_conversation_id),
    ]
    if web_search_enabled:
        tools.append(web_search)
    return tools
