"""Diagnose the local setup: Ollama server, installed models, LangSmith.

Run:  python scripts/check_setup.py
Exit code is 0 when Ollama is healthy, 1 otherwise (handy in scripts/CI).
"""

import sys

from localmind.config import configure_logging, get_settings
from localmind.llm import OllamaClient
from localmind.tracing import init_tracing


def main() -> int:
    settings = get_settings()
    configure_logging("WARNING")

    print(f"Ollama  @ {settings.ollama_base_url}")
    with OllamaClient(settings) as client:
        report = client.health()

    if report.reachable:
        print(f"  [ok] server running (version {report.version})")
        if report.installed_models:
            print("  [ok] installed models: " + ", ".join(report.installed_models))
        if report.missing_preferred:
            print("  [..] optional models not installed: " + ", ".join(report.missing_preferred))
            pulls = "; ".join(f"ollama pull {m}" for m in report.missing_preferred)
            print(f"       pull with: {pulls}")
    else:
        print("  [x] server not reachable")
    for hint in report.hints:
        print("  [!] " + hint.replace("\n", "\n      "))

    status = init_tracing(settings)
    print("\nLangSmith")
    if status.enabled:
        print(f"  [ok] tracing to project '{status.project}'")
    else:
        print(f"  [..] tracing disabled: {status.reason} (the app still works)")

    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
