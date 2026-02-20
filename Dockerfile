FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /usr/src/backend-ai

# ---- Install system dependencies (added ffmpeg for recording_manager.py) ----
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    libglib2.0-0 \
    libgobject-2.0-0 \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    ca-certificates \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# ---- Install uv ----
RUN pip install --no-cache-dir uv

# ---- Copy requirements ----
COPY requirements.txt .

# ---- Create venv & install deps ----
RUN uv venv /opt/venv \
    && uv pip install --python /opt/venv/bin/python -r requirements.txt

# RUN python agent.py download-files

ENV PATH="/opt/venv/bin:$PATH"

# ---- Copy app source ----
COPY . .

# ---- Download agent files ---- and run dev server ---- and useful to download files at build time for agent to work without internet access at runtime ----
RUN python agent.py download-files

EXPOSE 9000

CMD ["python", "agent.py", "dev"]
