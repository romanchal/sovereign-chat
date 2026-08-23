import sqlite3
import json
from datetime import datetime

class AuditLogger:
    def __init__(self, db_path="audit.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_traces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                model_used TEXT,
                prompt TEXT,
                tool_called TEXT,
                tool_input TEXT,
                tool_output TEXT,
                final_response TEXT
            )
        """)
        self.conn.commit()

    def log_trace(self, model: str, prompt: str, tool_called: str = None, tool_input: str = None, tool_output: str = None, final_response: str = None):
        self.cursor.execute("""
            INSERT INTO agent_traces 
            (timestamp, model_used, prompt, tool_called, tool_input, tool_output, final_response)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (datetime.utcnow().isoformat(), model, prompt, tool_called, json.dumps(tool_input) if tool_input else None, tool_output, final_response))
        self.conn.commit()
