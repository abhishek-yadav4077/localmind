"""Auto-generated conversation titles."""

from __future__ import annotations

import re

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from localmind.config import get_logger

logger = get_logger(__name__)

MAX_TITLE_CHARS = 60

TITLE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Write a short title (3 to 6 words) for a chat that starts with the exchange "
            "below. Output only the title: no quotes, no trailing punctuation, no prefix.",
        ),
        ("human", "User: {question}\n\nAssistant: {answer}"),
    ]
)


def fallback_title(question: str) -> str:
    """First words of the question; used when the model fails or rambles."""
    text = " ".join(question.split())
    return text if len(text) <= 40 else text[:40].rsplit(" ", 1)[0] + "…"


def clean_title(raw: str) -> str:
    """Normalize model output: first line, no 'Title:' prefix, quotes or markdown."""
    line = raw.strip().splitlines()[0] if raw.strip() else ""
    line = re.sub(r"^\s*(title\s*:\s*)", "", line, flags=re.IGNORECASE)
    line = line.strip(" \"'`*#.!:;")
    return line[:MAX_TITLE_CHARS].strip()


def generate_title(
    llm: BaseChatModel, question: str, answer: str, config: dict | None = None
) -> str:
    """Ask the model for a title; never raises (titles are cosmetic)."""
    chain = TITLE_PROMPT | llm | StrOutputParser()
    try:
        raw = chain.invoke(
            # Long answers add latency without improving the title.
            {"question": question[:1000], "answer": answer[:500]},
            config={"run_name": "generate_title", **(config or {})},
        )
        title = clean_title(raw)
    except Exception:
        logger.warning("title generation failed", exc_info=True)
        title = ""
    return title or fallback_title(question)
