from fastapi import FastAPI
import subprocess
import os

app = FastAPI()

@app.on_event("startup")
def start_agent():
    # Start your background AI agent
    subprocess.Popen(["python3", "agent.py", "dev"])

@app.get("/")
def health():
    return {"status": "AI Agent running 🚀"}
