# Kickoff prompt for Claude Code

`CLAUDE.md` in this folder is loaded automatically every session, so you do
**not** need to re-explain the project each time. Paste the block below as
your **first message** in a new Claude Code session in this directory.

---

```
Read CLAUDE.md first — it has the full project context, the hardware
constraints and the roadmap. Then read the existing code: models.yaml,
registry.py, router.py, ollama_client.py, egress.py, app.py and
static/index.html.

My situation right now:
- Windows 11, PowerShell only. No WSL, no Ubuntu, and I don't want to install
  either.
- Ollama is already installed natively on Windows and running.
- I have already pulled qwen3:8b and qwen2.5-coder:7b.
- I have not yet pulled a vision model or an embedding model.
- Nothing in this project has been run yet.

What I want from this first session, in order:

1. Walk me through getting Phase 0 running on Windows. Give me exact
   PowerShell commands. Flag any PowerShell-specific gotchas before I hit
   them — I know `curl` is aliased to Invoke-WebRequest and that the venv
   activation script can be blocked by execution policy, so warn me about
   anything else in that class.

2. Have me run check_gpu.ps1 and interpret the output with me. If
   `ollama ps` is not showing 100% GPU, fixing that comes before everything
   else.

3. Get the server up and confirm end to end: I type a question in the
   browser, tokens stream back, and the routing panel shows which model was
   chosen and why.

4. Then verify Phase 1's pass test with me: a coding question and a document
   question must visibly route to different models, and the swap timing must
   appear in the UI.

Rules I want you to hold me to:
- Nothing in this project may make an external network call. No CDN, no
  Google Fonts, no telemetry. Tell me if any dependency you suggest would.
- Don't fake outputs to make a demo look good.
- Don't start Phase 2 until Phase 1's pass test actually passes.

Ask me questions if something is ambiguous rather than guessing. Start by
telling me what you found in the code and what you think the first command
should be.
```

---

## For later sessions

Once Phase 0 and 1 are working, subsequent sessions can be much shorter,
because `CLAUDE.md` carries the context:

```
Read CLAUDE.md. Phase 1 is passing. Let's start Phase 2 — the agent loop
and the Docker sandbox. Plan it with me before writing any code.
```

Keep `CLAUDE.md` current as you go — tick off phases, record decisions and
note anything that surprised you about the hardware. It is the project's
memory across sessions.
