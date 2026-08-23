import httpx
import yaml
import time
import json
import re
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse

# Import our Phase 2 Arsenal
from tools import get_tools_schema, SandboxExecutor
from database import AuditLogger

app = FastAPI()
OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate" 
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"         

# 1. Initialize Registry, Sandbox, and Audit DB
with open("models.yaml", "r", encoding="utf-8") as f:
    registry = yaml.safe_load(f)["models"]

sandbox = SandboxExecutor()
audit = AuditLogger()

# 2. Intent Classifier (Tri-Model Routing)
def classify_intent(prompt: str):
    prompt_lower = prompt.lower()
    code_keywords = ["python", "code", "csv", "bug", "error", "def ", "class ", "json", "script", "sql", "function", "docker"]
    reasoning_keywords = ["analyze", "prove", "calculate", "derive", "architect", "step by step", "deep think", "strategy", "why does"]

    if any(kw in prompt_lower for kw in code_keywords):
        return "code", "Code / Engineering keywords detected", 0.95
    elif any(kw in prompt_lower for kw in reasoning_keywords):
        return "reasoning", "Deep reasoning / Logic keywords detected", 0.90
    else:
        return "chat", "General conversational query", 0.70

# 3. Swap Manager
class SwapManager:
    def __init__(self):
        self.current_model = None

    async def ensure_loaded(self, target_model: str):
        if self.current_model == target_model:
            return 0.0
            
        start_time = time.time()
        async with httpx.AsyncClient(timeout=60.0) as client:
            if self.current_model:
                await client.post(OLLAMA_GENERATE_URL, json={"model": self.current_model, "keep_alive": 0})
            await client.post(OLLAMA_GENERATE_URL, json={"model": target_model, "keep_alive": "5m"})
            
        self.current_model = target_model
        return round(time.time() - start_time, 2)

swap_manager = SwapManager()

# 4. Endpoints
@app.get("/")
async def serve_frontend():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.post("/chat")
async def chat_endpoint(request: Request):
    body = await request.json()
    prompt = body.get("prompt", "")
    
    role, reason, conf = classify_intent(prompt)
    target_config = next((m for m in registry if m["role"] == role), registry[0])
    model_name = target_config["name"]
    
    load_time = await swap_manager.ensure_loaded(model_name)
    
    async def proxy_stream():
        # 1. UI Metadata Badge
        meta = {"routing_meta": True, "model": model_name, "reason": reason, "confidence": conf, "load_time": load_time}
        yield json.dumps(meta) + "\n"
        
        # Format for Ollama's Chat API
        messages = [
            {"role": "system", "content": target_config["prompt_template"]},
            {"role": "user", "content": prompt}
        ]
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            
            # --- THE REACT AGENT LOOP ---
            if role == "code":
                MAX_ITERATIONS = 2
                
                for iteration in range(MAX_ITERATIONS):
                    yield json.dumps({"response": "\n_> ⚙️ Agent planning..._\n"}) + "\n"
                    
                    payload = {
                        "model": model_name,
                        "messages": messages,
                        "tools": get_tools_schema(),
                        "stream": False 
                    }
                    
                    r = await client.post(OLLAMA_CHAT_URL, json=payload)
                    msg = r.json().get("message", {})
                    
                    code_to_execute = None
                    t_name = None
                    used_native_tool = False
                    
                    # 1. Try Native Tool Call (The standard JSON API way)
                    if msg.get("tool_calls"):
                        tool_call = msg["tool_calls"][0]
                        t_name = tool_call["function"]["name"]
                        t_args = tool_call["function"]["arguments"]
                        code_to_execute = t_args.get("code", "")
                        used_native_tool = True
                        
                    # 2. The Markdown Interceptor (The violent regex fallback)
                    elif msg.get("content") and "```python" in msg.get("content"):
                        match = re.search(r'```python\n(.*?)\n```', msg["content"], re.DOTALL)
                        if match:
                            code_to_execute = match.group(1).strip()
                            t_name = "run_python (Regex Fallback)"
                            
                    # If it didn't call a tool OR write a python block, break the loop
                    if not code_to_execute:
                        break
                        
                    # Execute the code we captured
                    yield json.dumps({"response": f"\n_> 🛠️ Executing code in Sandbox:_\n```python\n{code_to_execute}\n```\n"}) + "\n"
                    
                    output = sandbox.run_python(code_to_execute)
                    
                    yield json.dumps({"response": f"\n_> 📋 Sandbox Output:_\n```text\n{output}\n```\n"}) + "\n"
                    
                    # Append history so the model observes the results before its final answer
                    if used_native_tool:
                        messages.append(msg)
                        messages.append({"role": "tool", "content": str(output)})
                    else:
                        messages.append({"role": "assistant", "content": msg.get("content", "")})
                        messages.append({"role": "user", "content": f"I executed your code in the sandbox. Here is the output:\n{output}\nNow provide the final summarized answer based on this output."})
                    
                    # Log the trace
                    audit.log_trace(model_name, prompt, t_name, code_to_execute, output, "Iterating...")

            # --- FINAL OUTPUT STREAMING ---
            payload = {
                "model": model_name,
                "messages": messages,
                "stream": True
            }
            req = client.build_request("POST", OLLAMA_CHAT_URL, json=payload)
            r = await client.send(req, stream=True)
            
            final_text = ""
            async for chunk in r.aiter_bytes():
                if chunk:
                    chunk_str = chunk.decode('utf-8')
                    for line in chunk_str.split('\n'):
                        if line.strip():
                            try:
                                data = json.loads(line)
                                if "message" in data and "content" in data["message"]:
                                    content = data["message"]["content"]
                                    final_text += content
                                    yield json.dumps({"response": content}) + "\n"
                            except json.JSONDecodeError:
                                pass
            
            # Save the final log
            audit.log_trace(model_name, prompt, None, None, None, final_text)

    return StreamingResponse(proxy_stream(), media_type="application/x-ndjson")