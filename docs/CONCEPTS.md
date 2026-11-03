# Concepts

Short answers to the questions this project is built around, each with a pointer
to where LocalMind uses the idea. Written to be explainable in an interview.

---

## 1. What is Ollama and how does it serve models locally?

Ollama is a local runtime for open-weight LLMs. It wraps the `llama.cpp`
inference engine behind a CLI and an HTTP server.

- **Pull**: `ollama pull llama3.2` downloads a model from the Ollama registry as
  content-addressed layers (weights in **GGUF** format, usually already
  quantized, plus the prompt template and default parameters) into `~/.ollama/models`.
- **Serve**: `ollama serve` (started automatically by the desktop app) listens on
  `localhost:11434`. The main endpoints are `/api/chat` (messages), `/api/generate`
  (raw prompt), `/api/embed` (embeddings), `/api/tags` (installed models) and
  `/api/pull`. It also exposes an OpenAI-compatible `/v1` API.
- **Load on demand**: the first request loads the model into RAM / VRAM (that's
  why the first message is slow). It stays loaded for `keep_alive` (5 minutes by
  default), then is unloaded. If the model doesn't fit in VRAM, some layers run on
  the GPU and the rest on the CPU.
- **Concurrency**: `OLLAMA_NUM_PARALLEL` and `OLLAMA_MAX_LOADED_MODELS` control how
  many requests and models are served at once.

**In LocalMind:** `llm/ollama_client.py` uses the REST API for health checks,
listing and pulling models; `ChatOllama` handles inference.

## 2. LLM wrapper vs chat model wrapper

| | `OllamaLLM` (LLM) | `ChatOllama` (chat model) |
|---|---|---|
| Input / output | string → string | list of messages → `AIMessage` |
| Ollama endpoint | `/api/generate` | `/api/chat` |
| Roles (system / user / assistant) | no, you build the prompt string | yes |
| Tool calling, structured output | no | yes (`bind_tools`, `with_structured_output`) |
| Token usage on the response | in `generation_info` | `usage_metadata` |

Chat models are LangChain's main interface today: almost every model is
trained as a chat model, and agents need roles and tool calls. The plain
LLM interface remains for completion-style use.

**In LocalMind:** `scripts/basic_chat.py` shows both side by side; the rest of
the app uses `ChatOllama` only, created in one place (`llm/factory.py`).

## 3. Local models vs API models

| | Local (Ollama) | Hosted API (OpenAI, Anthropic, ...) |
|---|---|---|
| **Privacy** | Prompts never leave the machine; works offline | Data goes to a third party (subject to their retention policy) |
| **Cost** | Hardware + electricity; no per-token bill | Pay per token; no hardware |
| **Latency** | No network round trip, but speed depends on your CPU/GPU; cold start when loading a model | Network latency, but very fast inference on datacenter GPUs |
| **Quality** | Small models (1–8B) are good at simple tasks, weaker at complex reasoning and long context | Frontier models are much stronger |
| **Operations** | You manage models, versions, memory | Nothing to run; rate limits and outages are theirs |

Rule of thumb: local for privacy-sensitive, offline, high-volume or simple tasks;
API models for hard reasoning or when quality matters most.

## 4. Quantization basics

Quantization stores model weights with fewer bits: 16-bit floats become 8-, 5- or
4-bit integers plus scaling factors.

- **Memory** ≈ parameters × bits / 8. A 3B model is ~6 GB at FP16 and ~2 GB at
  4-bit, which is what `ollama list` shows for `llama3.2`. The KV cache for the
  context window comes on top.
- **Speed**: generating each token reads all the weights from memory, so decoding
  is memory-bandwidth bound. Smaller weights mean more tokens per second.
- **Quality**: 4–5 bit "k-quants" such as `Q4_K_M` (mixed precision, more bits for
  sensitive layers) lose little quality. Below ~3 bits quality drops quickly.
- Ollama's default tags are usually `Q4_K_M`; other tags (e.g. `q8_0`) trade
  memory for quality. The Settings page shows each model's quantization.

## 5. Context window and memory strategies

The **context window** is the maximum number of tokens (prompt + answer) the
model can attend to. Two practical points:

- Ollama's runtime window is `num_ctx`, which by default is much smaller than
  what the model was trained for. Longer prompts are truncated *from the start*
  without an error, so the model seems to "forget". LocalMind always sets
  `num_ctx` explicitly (`llm/factory.py`).
- A larger window costs memory (the KV cache grows with it) and slows prompt processing.

Memory strategies for chat:

| Strategy | Idea | Trade-off |
|---|---|---|
| Full buffer | Send the whole history | Simple; overflows eventually |
| Sliding window | Last N turns only | Cheap; forgets early facts |
| Summary | Replace old turns with an LLM summary | Keeps gist; costs extra calls; lossy |
| **Summary + window** | Summary of old turns + last N verbatim | Good balance (**what LocalMind uses**) |
| Retrieval memory | Embed past turns, retrieve the relevant ones | Scales to long histories; more moving parts |

**In LocalMind:** `memory/summarizer.py` summarizes only when the prompt reaches
75% of the usable window (`num_ctx` minus room for the answer), keeps the last 6
messages verbatim, and marks the message with "memory summarized". Tokens are
estimated at ~4 characters each because Ollama has no tokenizer endpoint.

## 6. LangSmith: traces, runs, feedback, datasets, experiments

- **Run**: one step of execution: an LLM call, a chain, a tool call, a retriever.
  It records inputs, outputs, latency, token usage, errors, tags and metadata.
- **Trace**: the tree of runs for one request (for example agent → model → tool →
  model). It shows exactly what the model saw and why it answered as it did.
- **Project**: where traces are stored (`LANGSMITH_PROJECT=localmind`).
- **Tags and metadata**: labels for filtering. LocalMind tags every run with `mode`,
  `model` and `persona`, and adds `conversation_id` and `app_version` as metadata.
  `conversation_id` groups a chat into a **thread**.
- **Feedback**: a score attached to a run. Here: 👍/👎 from users
  (`user-score`), arena votes (`arena-win`), and evaluator scores.
- **Dataset**: examples (inputs plus reference outputs) to test against.
  `eval/dataset.json` becomes `localmind-eval-<hash>`.
- **Experiment**: one run of a target (a model) over a dataset, scored by
  evaluators. Several experiments can be compared side by side.

**Why observability matters in production:** LLM output is non-deterministic and
fails silently. It doesn't crash, it gives a wrong answer. Traces let you debug the
exact prompt; latency and token metrics catch slowdowns and cost growth; user
feedback finds bad answers; datasets and experiments turn "it seems better" into
a measured comparison and catch regressions before release.

## 7. LLM-as-judge: pros and cons

Using a model to grade another model's answers.

**Pros**
- Grades meaning, not exact strings: "Canberra is the capital" and "Canberra." both pass.
- Cheap and fast compared to human labelling; scales to many examples and models.
- Works for open-ended tasks (summaries, explanations) where exact match is impossible.

**Cons**
- **Biased**: judges prefer their own model's answers (self-preference), longer
  answers (verbosity) and the first option shown (position).
- **Unreliable when small**: a 3B local judge makes grading mistakes and is
  sensitive to prompt wording.
- **Not ground truth**: scores must be checked against human judgement before they
  are trusted.

**Mitigations used here:** a reference answer for every question, a short rubric
(1 / 0.5 / 0), JSON-schema structured output, temperature 0, a warning when the
judge is also a candidate, deterministic checks where possible (refusal regex,
word counts, timing), the judge model recorded with every run, and a "models
disagree" view on the Evaluation page for spot checks.
