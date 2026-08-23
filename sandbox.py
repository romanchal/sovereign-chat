"""
Docker sandbox — the agent's only path to executing generated code.

`--network=none` is the point. Even if the model writes `urllib.request.urlopen`,
the container has no route out. This is what lets the agent run untrusted
code without breaking the sovereignty claim.

Import-guarded: on machines without Docker (the build laptop), the module
still imports and `available` reports False, so the app can boot and the
non-agent paths keep working.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

try:
    import docker  # type: ignore
    from docker.errors import ContainerError, DockerException  # type: ignore
    _DOCKER_IMPORT_ERROR: str | None = None
except Exception as exc:  # pragma: no cover
    docker = None  # type: ignore
    ContainerError = DockerException = Exception  # type: ignore
    _DOCKER_IMPORT_ERROR = str(exc)


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
        self.client = None
        self.error: str | None = _DOCKER_IMPORT_ERROR
        if docker is None:
            return
        try:
            self.client = docker.from_env()
            self.client.ping()
        except Exception as exc:
            self.client = None
            self.error = str(exc)

    @property
    def available(self) -> bool:
        return self.client is not None

    def run_python(self, code: str) -> str:
        if not self.available:
            return f"[sandbox unavailable] {self.error or 'docker not connected'}"
        try:
            raw = self.client.containers.run(
                image=SANDBOX_IMAGE,
                command=["python", "-c", code],
                network_mode="none",
                mem_limit=SANDBOX_MEM,
                remove=True,
                stdout=True,
                stderr=True,
            )
            out = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
            return out.strip() or "[ok] no output"
        except ContainerError as exc:
            stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else str(exc.stderr)
            return f"[error] {stderr.strip()}"
        except Exception as exc:
            return f"[system error] {exc}"
