# Web MVP for RAG Chat

This file describes how to run the MVP web application without changing the
existing evaluation pipeline.

## 1) Install dependencies

```bash
pip install -r requirements.txt
pip install -r requirements-web.txt
```

## 2) Configure admin account (optional)

By default:
- username: `admin`
- password: `admin`

Override with environment variables:

```bash
set RAG_ADMIN_USERNAME=admin
set RAG_ADMIN_PASSWORD=strong_password
set RAG_ADMIN_TOKEN=custom_admin_token
set RAG_LLM_MODEL=qwen:7b
```

Also make sure Ollama is running and required models are available.

## 3) Run app

```bash
uvicorn webapp.app:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000).
React SPA view is available at [http://127.0.0.1:8000/chat](http://127.0.0.1:8000/chat) and [http://127.0.0.1:8000/spa](http://127.0.0.1:8000/spa).
Legacy server-rendered view is available at [http://127.0.0.1:8000/legacy](http://127.0.0.1:8000/legacy).

## What is included

- Reactive frontend on React + Ant Design (single page)
- Chat history persistence per browser session (cookie-based chat id)
- Sequential follow-up questions with recent dialogue context
- Chat reset button
- Fixed admin login for ingesting files
- File upload + indexing (`txt`, `md`, `csv`, `json`, `log`, `pdf`, `docx`)
- Source snippets shown with each answer
- Source content viewer by click (in chat and sidebar)

Data folders are created automatically in `webapp_data/`.

## Quick deploy for university (Docker)

This is the easiest way to run the app on a new machine.

### 1) Prerequisites

- Docker Desktop (or Docker Engine + Compose)
- Ollama running on host machine with required models pulled

### 2) Configure env

```bash
copy .env.example .env
```

Then edit `.env` and set secure admin credentials.

### 3) Build and run

```bash
docker compose up -d --build
```

### 4) Open app

- [http://127.0.0.1:8000](http://127.0.0.1:8000)

### 5) Stop app

```bash
docker compose down
```

Notes:
- Knowledge base files are persisted in local `webapp_data/` via volume mapping.
- By default the container uses `http://host.docker.internal:11434` to reach Ollama on host.
- If deploying on Linux server, replace `host.docker.internal` with actual Ollama host/IP.

## Full Docker mode (including Ollama)

If you want everything in Docker (web app + Ollama), use the full compose file.

### 1) Configure env

```bash
copy .env.example .env
```

Set at least:
- `RAG_ADMIN_USERNAME`
- `RAG_ADMIN_PASSWORD`
- `OLLAMA_MODEL` (for generation, default `qwen:7b`)
- `OLLAMA_EMBED_MODEL` (for embeddings, default `nomic-embed-text`)

### 2) Start full stack

```bash
docker compose -f docker-compose.full.yml up -d --build
```

This starts:
- `ollama` container
- one-time `ollama-init` container that pulls required models
- `rag-webapp` container connected to internal `ollama`

### 3) Open app

- [http://127.0.0.1:8000](http://127.0.0.1:8000)

### 4) Stop full stack

```bash
docker compose -f docker-compose.full.yml down
```

Notes:
- First startup may take significant time because models are downloaded.
- Ollama models are persisted in Docker volume `ollama_data`.
