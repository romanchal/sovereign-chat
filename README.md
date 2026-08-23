# Sovereign AI Workbench

An air-gapped, self-hosted agentic AI assistant for confidential industrial
work. Built for **Smart India Hackathon 2026**, problem statement
**SIH26117** (Mangalore Refinery and Petrochemicals Limited).

Nothing this system processes ever leaves the machine it runs on — and it
proves that live, rather than claiming it.

---

## Setup — Windows 11 + PowerShell

No WSL, no Ubuntu. Everything below runs in PowerShell.

### 1 · Prerequisites

- **NVIDIA driver R570 or newer.** The RTX 50-series (Blackwell) needs
  CUDA 12.8+. Check with `nvidia-smi`.
- **Ollama for Windows** — <https://ollama.com/download>
- **Python 3.11 or 3.12** — <https://python.org> (tick *Add to PATH*)

### 2 · Pull the models

```powershell
ollama pull qwen3:8b
ollama pull qwen2.5-coder:7b
ollama list
```

### 3 · Preflight — do this before anything else

```powershell
# If PowerShell refuses to run the script:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

.\check_gpu.ps1
```

You are looking for **100% GPU** and **30+ tokens/sec**. Anything less means
the model is spilling to CPU, and that is worth fixing before writing code.

### 4 · Install and run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>.

### 5 · Try the router

Ask a coding question, then a document question, and watch the routing panel:

> Write a Python function to parse an inspection CSV

> Summarise what a P&ID is for a new engineer

They must land on different models, and the swap timing must appear.

---

## PowerShell gotchas

| Problem | Fix |
|---|---|
| `curl` behaves oddly | PowerShell aliases it to `Invoke-WebRequest`. Use `curl.exe` or `Invoke-RestMethod`. |
| `Activate.ps1` blocked | `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| Ollama ignores env vars | The tray app starts at login and does not see `$env:` vars set in a shell. Set them in System Properties → Environment Variables, then quit and restart Ollama from the tray. |
| Model unloads mid-demo | Set `OLLAMA_KEEP_ALIVE=30m` the same way, or hit **Warm models** in the UI first. |

---

## Before you present

```powershell
# Warm every model so nothing loads cold on stage
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/warm -Method Post
```

Then unplug the ethernet cable and check the egress panel reads **0**.

---

## Project layout

```
models.yaml          the model registry — add models here, touch no Python
registry.py          loads and validates the registry
router.py            intent classification → model choice, with reasons
ollama_client.py     streaming, VRAM swap manager, warm-up
egress.py            the sovereignty proof
app.py               FastAPI application
static/index.html    single-file UI, zero external assets
check_gpu.ps1        Windows preflight
CLAUDE.md            project context for Claude Code
KICKOFF_PROMPT.md    what to paste into Claude Code first
```

## Licence

MIT for the code. Model weights carry their own upstream licences.
