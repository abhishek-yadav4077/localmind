# Screenshot checklist

Save each image here **with exactly this file name** (the main README links to them).
Capture from your own machine with real models; never use mock-ups.

**Before every capture:** hide API keys, emails and personal data; use a ~1400 px
wide window; light theme; crop browser chrome.

## LangSmith (Assignment 24, Task 1, step 6: also upload these to Drive)

| ✓ | File | What it must show | Where |
|---|------|-------------------|-------|
| ☐ | `langsmith-01-project.png` | The `localmind` project run list with several traces: names (`basic_chat`, `chat`, `agent`, `arena`, `rag`), latency, tokens, tags column | smith.langchain.com → Projects → `localmind` |
| ☐ | `langsmith-02-trace.png` | One `chat` trace opened: the prompt messages (persona system prompt + history), the output, latency, token counts, and the **Metadata** (`conversation_id`, `model`, `persona`, `mode`, `app_version`) and **Tags** panels | Click a `chat` run |
| ☐ | `langsmith-03-feedback.png` | A run's **Feedback** tab showing `user-score` (given with 👍/👎 in the UI) with your comment | Same run → Feedback |
| ☐ | `langsmith-04-dataset.png` | The `localmind-eval-<hash>` dataset with its 30 examples (inputs + reference outputs, category metadata) | Datasets & Experiments → dataset |
| ☐ | `langsmith-05-experiments.png` | The comparison view of the per-model experiments, scores side by side (`correctness`, `conciseness`, `refusal_correct`, `latency_s`) | "Compare in LangSmith" button on the Evaluation page |
| ☐ | `langsmith-06-basic-chat.png` | *(Assignment)* the trace of `python scripts/basic_chat.py` with tags `assignment-24`, `basic-chat` | Filter runs by tag `assignment-24` |

Optional but good: `langsmith-07-agent-trace.png` (agent → tools → agent loop with
tool inputs/outputs), `langsmith-08-thread.png` (Threads view of one conversation),
`langsmith-09-arena-feedback.png` (runs filtered by `battle_id` with `arena-win` scores).

## App

| ✓ | File | What it must show |
|---|------|-------------------|
| ☐ | `app-chat.png` | A conversation with a persona, streaming answer finished, the stats line (latency · TTFT · tok/s · tokens), 👍 selected, sidebar with conversations |
| ☐ | `app-arena.png` | 3 models side by side **after** voting (🏆 on the winner, stats visible) + leaderboard below |
| ☐ | `app-agent.png` | An agent answer with the "🛠 N tool calls" expander open (calculator + date) |
| ☐ | `app-documents.png` | A PDF answer with `[n]` citations and the sources expander open |
| ☐ | `app-dashboard.png` | Dashboard after real use: summary numbers + per-model charts |
| ☐ | `app-evaluation.png` | Evaluation page of a full run: summary table + correctness-by-category chart |
| ☐ | `app-settings.png` | Installed models table + LangSmith status |
| ☐ | `demo.gif` | 20–40 s: ask a question in Chat → stats appear → 👍 → open trace in LangSmith. Record with [ScreenToGif](https://www.screentogif.com/) (Windows), [Kap](https://getkap.co/) (macOS) or [Peek](https://github.com/phw/peek) (Linux); keep it under ~8 MB |

When everything is captured, every image link in the main README resolves.
