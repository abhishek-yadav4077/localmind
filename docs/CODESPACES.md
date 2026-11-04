# Run LocalMind in your browser (GitHub Codespaces)

No installation on your computer. A Codespace is a Linux machine in the cloud,
built from this repository, that you use through your browser. Everything (Python,
Ollama, the models, the app) runs there.

**Cost:** free GitHub accounts include a monthly allowance (about 60 hours on the
default 2-core / 8 GB machine, plus 15 GB of storage). Personal accounts have a
$0 spending limit by default, so you can't be charged: the Codespace just stops when the
allowance runs out. Check usage at <https://github.com/settings/billing>.

> **Speed:** the models run on the Codespace's CPU (no GPU), so answers take a few
> seconds to start and stream slowly. That's expected, and the speed numbers you
> record will honestly say "2 cores, no GPU". For faster runs, choose the 4-core
> machine in step 2 (it uses the free allowance twice as fast).

---

## 1. (Optional) Add your LangSmith key as a Codespaces secret

Do this **before** creating the Codespace so tracing works from the start.

1. Get a key: <https://smith.langchain.com> → Settings (gear) → API Keys → Create API Key.
2. Open <https://github.com/settings/codespaces> → **Codespaces secrets** → **New secret**.
3. Name: `LANGSMITH_API_KEY` · Value: your key · Repository access: `abhishek-yadav4077/localmind`.
4. Click **Add secret**.

The key never goes into the code or into Git. If you add it after creating the
Codespace, stop and start the Codespace once so it picks the key up.

## 2. Create the Codespace

1. Open <https://github.com/abhishek-yadav4077/localmind>.
2. Click the green **Code** button → **Codespaces** tab → **Create codespace on main**.
   (For the 4-core machine: click **⋯** → **New with options** → Machine type.)
3. A VS Code editor opens in your browser. The **first** start runs
   `.devcontainer/setup.sh` automatically: it installs the Python packages,
   installs Ollama and downloads ~5.5 GB of models. **This takes about 10–20
   minutes.** Watch the progress in the terminal panel ("Creation log"). Wait until it says
   `Setup complete`.

Later starts take under a minute; everything stays installed.

## 3. Check that everything works

Open a terminal if none is open (menu ☰ → Terminal → New Terminal) and run:
```bash
python scripts/check_setup.py
```
Expect `[ok]` for Ollama and its models, and `[ok] tracing to project 'localmind'` if you added the secret.

Run the test suite (no models needed):
```bash
pytest
```

## 4. The assignment script

```bash
python scripts/basic_chat.py
```
The first answer is slow while the model loads. Then open
<https://smith.langchain.com> → Projects → `localmind` to see the `basic_chat` traces.

## 5. Start the app

```bash
bash .devcontainer/run-app.sh
```
When it says `Starting UI`, a notification offers to open port 8501. Click
**Open in Browser**. If you miss it: open the **Ports** tab (next to Terminal) →
hover over port **8501** → click the 🌐 globe icon. The app opens in a new browser tab.

The API's interactive docs are on port **8000** at `/docs`.

To stop the app: click in the terminal and press **Ctrl+C**.

### Tour (creates the data for screenshots and the dashboard)
1. **Chat:** in the sidebar, pick `llama3.2` and the **Tutor** persona, ask a
   question, then click 👍, add a comment, and click "🔍 LangSmith trace".
2. **Model Arena:** pick `llama3.2:latest` and `llama3.2:1b`, turn **Sequential
   ON** (8 GB RAM), Fight, vote. Do 3–5 battles.
3. **Agent:** ask *"What is 17.5% of 2340, and what weekday is it today?"* and open the tool calls.
4. **Documents:** upload a small text-based PDF → Index files → ask about it.
5. **Dashboard** and **Settings:** look around.

## 6. Screenshots

Use your computer's screenshot tool (Windows: **Win + Shift + S**). File names
and what each must show: [`docs/screenshots/README.md`](screenshots/README.md).

To put them in the repo: in the Codespace's file Explorer (left), open
`docs/screenshots`, then **drag the image files from your computer onto that folder**.

## 7. Evaluation (real benchmark numbers)

Stop the app first (Ctrl+C) so the models have all the CPU and memory.

Run the evaluation **in the background with a log file**. The browser terminal can
freeze on long-running output; a background job keeps going regardless, and even
survives closing the terminal. (`-u` writes each line to the log immediately.)

**Quick test** (5 questions per model):
```bash
nohup python -u scripts/run_eval.py --models llama3.2 llama3.2:1b --judge qwen2.5:3b --per-category 1 > /tmp/eval.log 2>&1 &
```
Watch it (Ctrl+C only stops *watching*, not the evaluation):
```bash
tail -f /tmp/eval.log
```
Other useful checks: `pgrep -af run_eval` (is it still running?) and `ollama ps`
(which models are working). It's finished when the log shows `Done in … min`, a
results table and a `LangSmith comparison:` link.

**Full run** (30 questions per model), same way:
```bash
nohup python -u scripts/run_eval.py --models llama3.2 llama3.2:1b --judge qwen2.5:3b > /tmp/eval-full.log 2>&1 &
```
```bash
tail -f /tmp/eval-full.log
```
Then write the real results (and the Codespace's hardware) into the README:
```bash
python scripts/benchmark_table.py --write
```

> A Codespace stops after its **idle timeout** (30 minutes by default) without
> activity in the editor. Raise it at <https://github.com/settings/codespaces>
> (e.g. 240 minutes) before a long run, and keep the tab open.

## 8. Save your work to GitHub

The Codespace is already logged in to your GitHub account:
```bash
git add .
```
```bash
git commit -m "docs: add screenshots and benchmark results"
```
```bash
git push
```
(Or use the **Source Control** panel on the left: type a message → Commit → Sync.)
`.env` is ignored by Git, so no keys are uploaded.

## 9. Stop the Codespace when you're done

GitHub → your profile picture → **Your codespaces** → **⋯** → **Stop codespace**.
When you have everything you need, **Delete** it to free storage (the code is safe
on GitHub; you can always create a new one).

---

### Troubleshooting

| Problem | Fix |
|---|---|
| A command shows nothing at all (not even its first line) and Ctrl+C prints no error | The terminal itself is stuck, not the app: close it (🗑 trash icon in the terminal list) and open a new one with **+** |
| `Could not reach Ollama` | `bash .devcontainer/start-ollama.sh` (Ollama doesn't restart by itself after the Codespace sleeps) |
| A model is missing | `ollama pull <model>` |
| Setup failed during creation | Run `bash .devcontainer/setup.sh` again in the terminal; it skips finished steps |
| LangSmith shows "tracing disabled" | Add the `LANGSMITH_API_KEY` secret (step 1), then stop and start the Codespace |
| Port 8501 shows an error page | The app isn't running: start `bash .devcontainer/run-app.sh` |
| Very slow or "killed" | 8 GB is tight: use one model at a time, Arena in Sequential mode, or the 4-core machine |
