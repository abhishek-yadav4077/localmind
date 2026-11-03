from localmind.chat.stats import StreamTimer, compute_stats
from localmind.chat.titles import clean_title, fallback_title, generate_title
from tests.fakes import FakeOllamaChat


def timer(start=0.0, first=0.2, end=1.2) -> StreamTimer:
    return StreamTimer(start=start, first_token=first, end=end)


def test_tokens_per_sec_prefers_ollama_eval_duration() -> None:
    stats = compute_stats(
        timer(),
        {"eval_count": 50, "eval_duration": 2_000_000_000},
        {"input_tokens": 10, "output_tokens": 50},
    )
    assert stats.tokens_per_sec == 25.0
    assert stats.latency_ms == 1200.0
    assert stats.ttft_ms == 200.0
    assert (stats.prompt_tokens, stats.completion_tokens) == (10, 50)


def test_tokens_per_sec_falls_back_to_wall_clock() -> None:
    stats = compute_stats(timer(), {}, {"input_tokens": 3, "output_tokens": 30})
    assert stats.tokens_per_sec == 30.0  # 30 tokens over 1.0 s after first token


def test_missing_data_is_none_not_invented() -> None:
    stats = compute_stats(StreamTimer(start=0.0, end=1.0), None, None)
    assert stats.tokens_per_sec is None and stats.ttft_ms is None
    assert stats.completion_tokens is None


def test_clean_title() -> None:
    assert clean_title('Title: "Learning Python Basics".') == "Learning Python Basics"
    assert clean_title("**Debugging SQL**\nextra line") == "Debugging SQL"
    assert clean_title("") == ""


def test_fallback_title_truncates_on_word_boundary() -> None:
    assert fallback_title("short question") == "short question"
    long = "please explain the difference between processes and threads in detail"
    assert fallback_title(long).endswith("…") and len(fallback_title(long)) <= 41


def test_generate_title_falls_back_on_error() -> None:
    llm = FakeOllamaChat(error=RuntimeError("boom"))
    assert generate_title(llm, "what is a vector database", "...") == "what is a vector database"
