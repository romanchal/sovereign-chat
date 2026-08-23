import docker
from pydantic import BaseModel, Field
import json

class RunPythonSchema(BaseModel):
    code: str = Field(description="Python code to execute in the secure sandbox")

def get_tools_schema():
    return [{
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "Execute python code in an isolated, air-gapped Docker container. Returns stdout or stderr.",
            "parameters": RunPythonSchema.model_json_schema()
        }
    }]

class SandboxExecutor:
    def __init__(self):
        try:
            self.client = docker.from_env()
        except Exception as e:
            print(f"Failed to connect to Docker: {e}")

    def run_python(self, code: str) -> str:
        try:
            output = self.client.containers.run(
                image="python:3.12-slim",
                command=["python", "-c", code],
                network_mode="none",
                mem_limit="256m",
                remove=True,
                capture_output=True,
                text=True
            )
            return output.strip() if output else "Execution successful, no output."
        except docker.errors.ContainerError as e:
            return f"Error:\n{e.stderr}"
        except Exception as e:
            return f"System Error:\n{str(e)}"
