# Sovereign AI Workbench — Project Context

> Claude Code reads this file automatically at the start of every session.
> It is the single source of truth for what this project is and what has
> already been decided. Keep it updated as phases complete.

---

## 1. What this is

A submission for **Smart India Hackathon 2026**, problem statement
**SIH26117**, posed by **Mangalore Refinery and Petrochemicals Limited (MRPL)**.

> *Sovereign On-Premise Agentic AI Workbench using Open-Weight Multimodal
> LLMs for Confidential Industrial Work*
> Category: Software · Theme: Smart Automation · Submission closes **20 Sep 2026**

**The problem in one line:** MRPL wants ChatGPT, but locked inside their own
building with the internet cable cut.

Refineries and PSUs generate large volumes of sensitive knowledge work —
approval notes, board presentations, engineering calculations, scanned
inspection reports, P&IDs, vendor negotiations. None of it may go to a cloud
AI. So today staff either do the work manually, or quietly paste confidential
material into public tools anyway. MRPL says this explicitly in the problem
statement. We are removing that choice.

**Immediate goal:** be selected at the college internal round.
**Stretch goal:** place at the national level.

---

## 2. What MRPL explicitly requires

Every one of these is written into the official problem statement text.
Missing any is a merit failure.

1. **Multi-model serving with automatic selection.** Several open-weight
   models available at once; the system picks the right one per task.
   Must be demoed across **at least two different task types**.
2. **Pluggable model registry.** New models addable later *without
   redesigning the system*.
3. **Genuinely agentic.** Plans multi-step work, calls local tools (file
   read/write, sandboxed code execution, spreadsheet work, internal document
   search), and **iterates** rather than answering once and stopping.
4. **Multimodal.** Scanned PDFs, handwritten notes, engineering drawings and
   photographs, read via on-device OCR and vision models.
5. **Real deliverables.** Word / Excel / PowerPoint files, working code,
   calculations with steps shown — not chat replies.
6. **Local knowledge grounding.** Answers grounded in the organisation's own
   manuals, SOPs and past correspondence. Nothing external.
7. **Proof of the air gap.** MRPL's exact framing: the system must show,
   through logs or a visible network monitor, that **no external calls are
   made at any point** — *"That's the actual proof of the sovereign claim,
   not just a statement of it."*

**Requirement 7 is the differentiator.** Most competing teams will claim
sovereignty on a slide. We prove it live. Treat it as a first-class feature,
not a finishing touch.

### Two concessions MRPL grants us

- A **smaller open-weight model is explicitly permitted** if venue hardware
  cannot run a large one.
- **No proprietary data is required.** Public sample scanned PDFs and public
  sample P&IDs are acceptable for the demo.

---

## 3. Hardware — this shapes every technical decision

Target machine (the developer's own laptop):

| | |
|---|---|
| CPU | AMD Ryzen 7 260 · 8 cores / 16 threads |
| RAM | 24 GB |
| GPU | **NVIDIA RTX 5050 · 8 GB VRAM** (Blackwell, `sm_120`) |
| Storage | 1 TB NVMe SSD |
| OS | **Windows 11 Home** |
| Shell | **Windows PowerShell — no WSL2, no Ubuntu** |

### Decisions already made because of this hardware

**Use Ollama, not vLLM.** Blackwell requires CUDA 12.8+ and driver R570+.
Ollama ships its own CUDA runtime and sidesteps the entire toolchain problem.
vLLM on Windows is not worth the days it would cost. Ollama is installed
natively on Windows and is reachable at `http://127.0.0.1:11434`.

**Stay on native Windows. Do not suggest installing WSL2/Ubuntu.**
Everything needed works in PowerShell. WSL2 would also *weaken* the air-gap
demo, because its NAT layer muddies the "no external calls" story. Docker
Desktop may be installed later purely as sandbox plumbing (it manages its own
WSL backend invisibly) — the developer never opens an Ubuntu terminal.

**Models swap; they are not co-resident.** ~7.2 GB is usable after Windows
takes its share. The reasoning model alone is 5.5 GB. Therefore:

- Exactly one large model is resident at a time.
- The tiny embedding model stays resident permanently so retrieval never
  triggers a swap.
- A cold load costs roughly 3–6 s from NVMe. **Make the swap visible in the
  UI** (`unloaded coder → loaded vision in 4.1 s`) so it reads as deliberate
  orchestration rather than a hang.
- Always pre-warm every model before presenting.

**Push everything non-LLM onto the CPU.** 16 threads and 24 GB of RAM are
plentiful. Reranker, OCR, vector store, Docker sandbox and FastAPI all run on
CPU so the GPU is never competing with the application.

### Model roster

| Role | Ollama tag | VRAM | Status |
|---|---|---|---|
| Reasoning / writing | `qwen3:8b` | ~5.5 GB | **pulled** |
| Code | `qwen2.5-coder:7b` | ~4.5 GB | **pulled** |
| Vision / scans | `qwen2.5vl:7b` | ~6.6 GB | not yet pulled |
| Embeddings | `nomic-embed-text` | ~0.3 GB | not yet pulled |
| Reranker | `bge-reranker-base` | CPU | later |
| OCR | PaddleOCR | CPU | later |

With `qwen3` + `qwen2.5-coder` already present, MRPL's stated minimum of
auto-selection across two task types can be satisfied today.

---

## 4. Architecture

Five layers inside one hard boundary. Requests flow down; deliverables flow
back up. The only path that crosses the boundary is blocked and counted.

```
        ┌──────────────────────────────────────────────┐
        │  AIR-GAPPED — organisation's own GPU server   │
        │                                              │
        │  1 · Workbench UI                            │
        │      chat · artifacts · trace · EGRESS METER │
        │                  ↓ user task                 │
        │  2 · Model Router                            │
        │      classify → pick model → show why        │
        │      registry file = add models, no redesign │
        │                  ↓ routed request            │
        │  3 · Agent Orchestrator                      │
        │      plan → act → observe → verify → repeat  │
        │      every step to append-only audit log     │
        │                  ↓ tool calls                │
        │  4a · Tool Sandbox    │  4b · Knowledge Base │
        │      docker           │      SOPs, P&ID graph│
        │      --network=none   │      vector+BM25+rank│
        │                  ↓ inference                 │
        │  5 · Ollama — one large model resident       │
        │                                              │
        └──────────────────────────────────────────────┘
                     ╳  blocked, counted, displayed
                  Public cloud AI / Internet
```

### Current state of the code

Phase 0 + most of Phase 1 exist:

| File | Purpose |
|---|---|
| `models.yaml` | The registry. **Adding a model = editing this file only.** |
| `registry.py` | Loads and validates `models.yaml` |
| `router.py` | Rule-based intent classification → model choice + reason |
| `ollama_client.py` | Streaming client, swap manager, warm-up, `ollama ps` |
| `egress.py` | The sovereignty proof — counts external TCP connections |
| `app.py` | FastAPI: `/api/chat` (NDJSON stream), `/api/egress`, `/api/warm` |
| `static/index.html` | Single-file UI, **zero external assets** |
| `check_gpu.ps1` | PowerShell preflight — run before touching anything else |

---

## 5. Non-negotiable rules

**1. No external network calls. Anywhere. Ever.**
This is the entire thesis of the project. That means:
- No CDN scripts, no Google Fonts, no remote stylesheets in the UI.
  System fonts and inline CSS/JS only.
- No cloud AI APIs, no telemetry, no analytics, no crash reporting.
- Before adding any dependency, consider whether it phones home.
- The outbound firewall stays on during development, not just at demo time.
  A hidden call discovered the night before the demo is fatal.

**2. Never fake it.** If a model cannot do something, the system says so.
No hardcoded demo answers, no mock responses dressed up as real output.
Judges probe, and getting caught ends the run.

**3. Build for the demo, verify by the pass test.** Each phase below has a
test. Do not begin the next phase until the current one passes.

**4. A working core beats a broken feature list.** Five reliable features
beat fifteen half-built ones. Judges score what runs in front of them.

---

## 6. Roadmap

Submission closes **20 September 2026**.

| Phase | Goal | Pass test |
|---|---|---|
| **0 · Skeleton** | FastAPI + Ollama streaming into a browser | Tokens appear in a browser with Wi-Fi off |
| **1 · Router** | Registry, intent classification, swap manager, visible decision | Code question and doc question route to different models; a 4th model needs no code change |
| **2 · Agent** | Tool schemas, plan→act→observe→verify loop, Docker sandbox, audit log | "Write and test a CSV parser" completes alone, including recovering from one failure |
| **3 · Documents & vision** | OCR ingest, hybrid retrieval + rerank, VL model on scans/drawings, citations | Answers a question from a scanned PDF and names the page |
| **4 · Deliverables & proof** | docx/xlsx/pptx generation, live egress meter, two-role access control | A generated .docx opens in Word; egress counter never left zero |
| **5 · Rehearse** | No new features. Pre-warm script, backup video, submission writeup | Full demo runs with ethernet unplugged, no terminal touched |

### The five-minute demo this all builds toward

1. **0:00** Unplug the ethernet cable, physically, in front of the panel.
2. **0:30** Coding question → router picks coder model → code runs in sandbox, tests pass.
3. **1:30** Scanned inspection report → vision model reads it, including handwriting → drafts an approval note → downloadable `.docx`. *(This is MRPL's own worked example.)*
4. **3:00** Upload a P&ID → ask which valves sit on a given line → answered from the drawing.
5. **4:00** Log in as a junior engineer → same confidential question → declined, with reason.
6. **4:30** Add a model to the registry live → it appears in the router.
7. **4:45** Turn to the egress monitor: zero. Audit trail: complete. Cable: still unplugged.

---

## 7. Deliberately deferred

Real differentiators, but **post-selection** work. Attempting them before the
college round costs the reliable core.

- P&ID → queryable graph (strongest differentiator, but a project in itself)
- Fine-tuned YOLO symbol detector (needs a labelled dataset we don't have)
- Voice in/out (two more models, and the VRAM budget has no spare)
- Planner + critic agents (doubles inference while models are swapping)
- Drawing revision diff (finale material)

---

## 8. Working preferences

- Plain Python with Pydantic-typed tool schemas. **No heavy agent framework** —
  this gets debugged at 3 a.m. and every line must be readable.
- Explain *why*, not just *what*, when proposing a change.
- Flag honestly when something will not work on 8 GB of VRAM.
- Prefer editing existing files over creating new ones.
- Keep the UI dependency-free; it must open from `file://` if it has to.
