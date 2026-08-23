"""
Audit log — append-only SQLite trail of every agent action.

Sovereignty demo talking point: the panel can inspect audit.db after the run
and see every tool call, every input, every output. Nothing hidden.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).parent / "audit.db"


class AuditLogger:
    def __init__(self, db_path: Path = DB_PATH) -> None:
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS agent_traces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                model TEXT NOT NULL,
                prompt TEXT NOT NULL,
                event TEXT NOT NULL,
                payload TEXT
            )
        """)
        self.conn.commit()

    def log(self, model: str, prompt: str, event: str, payload: Any = None) -> None:
        self.conn.execute(
            "INSERT INTO agent_traces (ts, model, prompt, event, payload) VALUES (?,?,?,?,?)",
            (
                datetime.now(timezone.utc).isoformat(),
                model,
                prompt,
                event,
                json.dumps(payload, default=str) if payload is not None else None,
            ),
        )
        self.conn.commit()
