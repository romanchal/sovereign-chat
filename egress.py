"""
Egress monitor — the sovereignty proof.

MRPL's problem statement says the proof of the air gap must be visible
(logs or a network monitor showing no external calls), not merely claimed.

Counts ESTABLISHED TCP connections to non-private, non-loopback addresses,
FILTERED to processes that belong to the workbench (this Python process,
its children, and Ollama). Machine-wide traffic from Chrome, Edge WebView,
svchost etc. is out of scope for the sovereignty claim — the claim is that
*this workbench* makes no external calls, not that the host OS is offline.
"""
from __future__ import annotations

import ipaddress
import os
import time
from typing import Any

import psutil

_STARTED = time.time()
_PEAK_EXTERNAL = 0
_OWN_PID = os.getpid()

# Process names whose external connections count against the workbench.
# Lower-case, no extension check — psutil returns names like "ollama.exe".
_APP_PROC_NAMES = {
    "python", "python.exe", "pythonw.exe",
    "uvicorn", "uvicorn.exe",
    "ollama", "ollama.exe", "ollama app.exe",
    "ollama_llama_server", "ollama_llama_server.exe",
    "docker", "docker.exe", "com.docker.backend.exe",
}


def _own_pids() -> set[int]:
    pids = {_OWN_PID}
    try:
        me = psutil.Process(_OWN_PID)
        for child in me.children(recursive=True):
            pids.add(child.pid)
    except Exception:
        pass
    return pids


def _app_pids() -> set[int]:
    pids = _own_pids()
    for p in psutil.process_iter(["pid", "name"]):
        try:
            name = (p.info.get("name") or "").lower()
            if name in _APP_PROC_NAMES:
                pids.add(p.info["pid"])
        except Exception:
            continue
    return pids


def _is_external(addr: str) -> bool:
    if not addr:
        return False
    try:
        ip = ipaddress.ip_address(addr)
    except ValueError:
        return False
    return not (
        ip.is_loopback
        or ip.is_private
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_unspecified
        or ip.is_reserved
    )


def snapshot() -> dict[str, Any]:
    global _PEAK_EXTERNAL

    external: list[dict[str, Any]] = []
    inspected = 0
    degraded = False

    scope_pids = _app_pids()

    try:
        conns = psutil.net_connections(kind="inet")
    except (psutil.AccessDenied, PermissionError):
        conns = []
        degraded = True

    for c in conns:
        if c.status != psutil.CONN_ESTABLISHED or not c.raddr:
            continue
        # Only inspect connections owned by workbench processes.
        if c.pid not in scope_pids:
            continue
        inspected += 1
        host = c.raddr.ip if hasattr(c.raddr, "ip") else c.raddr[0]
        if not _is_external(host):
            continue

        proc = "unknown"
        try:
            if c.pid:
                proc = psutil.Process(c.pid).name()
        except Exception:
            pass

        external.append({
            "remote": f"{host}:{c.raddr.port if hasattr(c.raddr, 'port') else c.raddr[1]}",
            "pid": c.pid,
            "process": proc,
        })

    _PEAK_EXTERNAL = max(_PEAK_EXTERNAL, len(external))

    return {
        "external_count": len(external),
        "external": external[:20],
        "established_total": inspected,
        "peak_external": _PEAK_EXTERNAL,
        "clean": len(external) == 0,
        "never_dirty": _PEAK_EXTERNAL == 0,
        "uptime_s": int(time.time() - _STARTED),
        "degraded": degraded,
    }
