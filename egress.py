"""
Egress monitor — the sovereignty proof.

MRPL's problem statement says the proof of the air gap must be visible
(logs or a network monitor showing no external calls), not merely claimed.
This module is that proof, and it is why you build it in week one: if
something in your stack quietly phones home, you want to know now.

Counts ESTABLISHED TCP connections to non-private, non-loopback addresses.
Anything above zero while the workbench is running deserves investigation.
"""
from __future__ import annotations

import ipaddress
import time
from typing import Any

import psutil

# Populated once at import so the UI can show an uptime-clean streak.
_STARTED = time.time()
_PEAK_EXTERNAL = 0


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

    try:
        conns = psutil.net_connections(kind="inet")
    except (psutil.AccessDenied, PermissionError):
        # On Windows some system sockets need elevation. Report honestly
        # rather than showing a falsely clean zero.
        conns = []
        degraded = True

    for c in conns:
        if c.status != psutil.CONN_ESTABLISHED or not c.raddr:
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
