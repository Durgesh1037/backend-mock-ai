from fastapi import FastAPI
import subprocess
import threading
import sys

app = FastAPI()

def stream_logs(process):
    for line in process.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()

@app.on_event("startup")
def start_agent():
    print("🚀 Starting LiveKit Agent subprocess...")
    process = subprocess.Popen(
        ["python3", "agent.py", "start"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    threading.Thread(target=stream_logs, args=(process,), daemon=True).start()

@app.get("/")
def health():
    return {"status": "Wrapper running"}
