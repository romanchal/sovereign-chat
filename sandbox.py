"""
Docker sandbox — the agent's only path to executing generated code.

Uses the `docker` CLI via subprocess rather than the `docker` Python SDK
so we avoid the Windows named-pipe dependency (pywin32/pypiwin32), which
breaks on Python 3.14 today. The CLI ships with Docker Desktop and is
the same tool the user already relies on.

`--network=none` is the whole point. Even if the model writes
`urllib.request.urlopen`, the container has no route out. That is what
lets us run untrusted code without breaking the sovereignty claim.
"""
from __future__ import annotations

import shutil
import subprocess
from typing import Any

from pydantic import BaseModel, Field

SANDBOX_IMAGE = "python:3.12-slim"
SANDBOX_MEM = "256m"
SANDBOX_TIMEOUT_S = 20


class RunPythonSchema(BaseModel):
    code: str = Field(description="Python code to execute in the sandbox")


def get_tools_schema() -> list[dict[str, Any]]:
    return [{
        "type": "function",
        "function": {
            "name": "run_python",
            "description": (
                "Execute Python code in an isolated, air-gapped Docker container. "
                "Returns combined stdout/stderr. No network. Use for calculations, "
                "parsing, and verifying your own code."
            ),
            "parameters": RunPythonSchema.model_json_schema(),
        },
    }]


class SandboxExecutor:
    def __init__(self) -> None:
        self.docker_bin: str | None = shutil.which("docker")
        self.error: str | None = None
        if not self.docker_bin:
            self.error = "docker CLI not on PATH"
            return
        try:
            r = subprocess.run(
                [self.docker_bin, "info", "--format", "{{.ServerVersion}}"],
                capture_output=True, text=True, timeout=5,
            )
            if r.returncode != 0:
                self.error = (r.stderr or r.stdout or "docker info failed").strip()
        except Exception as exc:
            self.error = f"{exc}"

    @property
    def available(self) -> bool:
        return self.docker_bin is not None and self.error is None

    def run_python(self, code: str) -> str:
        if not self.available:
            return f"[sandbox unavailable] {self.error or 'docker not connected'}"
        try:
            r = subprocess.run(
                [
                    self.docker_bin, "run", "--rm",
                    "--network=none",
                    "--memory", SANDBOX_MEM,
                    SANDBOX_IMAGE,
                    "python", "-c", code,
                ],
                capture_output=True, text=True, timeout=SANDBOX_TIMEOUT_S,
            )
            out = (r.stdout or "") + (r.stderr or "")
            out = out.strip()
            if r.returncode != 0:
                return f"[error rc={r.returncode}] {out or 'no output'}"
            return out or "[ok] no output"
        except subprocess.TimeoutExpired:
            return f"[timeout] exceeded {SANDBOX_TIMEOUT_S}s"
        except Exception as exc:
            return f"[system error] {exc}"
