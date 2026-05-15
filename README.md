# Deployable Agentic RAG

This project now includes a local web app with:

- document upload for `.pdf` and `.txt`
- stored file list
- delete stored embeddings and uploaded files
- re-embed previously uploaded files
- chatbot over the ingested knowledge base
- persistent local Chroma storage

## Local Run

### 1. Install dependencies

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

Create `.env` from `.env.example` and set a real NVIDIA API key:

```text
NVIDIA_API_KEY=your_real_nvidia_api_key
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_MODEL=meta/llama-3.1-70b-instruct
```

### 3. Start the web app

```bash
uvicorn webapp:app --reload
```

Open `http://127.0.0.1:8000`

## What the Web App Supports

- Upload and embed new documents
- View every stored file in the UI
- Re-embed an uploaded file
- Delete embeddings and remove the uploaded file
- Ask chatbot questions against the stored knowledge base
- Clear saved chat memory

## Existing CLI Still Works

```bash
python main.py
python ingest.py data/sample.txt
```

## Project Files

- `webapp.py`: FastAPI app and browser UI
- `ingest.py`: extraction, chunking, embeddings, document management helpers
- `retriever.py`: vector retrieval and source-aware filtering
- `agent.py`: LLM chat logic and tool-driven retrieval
- `chroma_db/`: persistent vector database
- `uploads/`: uploaded source documents

## Deploy Online

The simplest deployment path is Render using a persistent disk.

### Render Setup

1. Push this project to GitHub.
2. In Render, create a new `Web Service`.
3. Connect the GitHub repo.
4. Use either:

Build command:

```bash
pip install -r requirements.txt
```

Start command:

```bash
uvicorn webapp:app --host 0.0.0.0 --port $PORT
```

Or deploy with the included `Dockerfile`.

### Render Environment Variables

Set:

```text
NVIDIA_API_KEY=your_real_nvidia_api_key
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_MODEL=meta/llama-3.1-70b-instruct
CHROMA_PERSIST_DIR=/app/data/chroma_db
UPLOAD_DIR=/app/data/uploads
MEMORY_FILE=/app/data/chat_memory.json
```

### Persistent Storage

Add one persistent disk and mount it to:

```text
/app/data
```

## Docker Run

```bash
docker build -t deployable-rag .
docker run -p 8000:8000 --env-file .env deployable-rag
```

Open `http://localhost:8000`

## Notes

- First embedding run may take longer because the sentence-transformer model downloads locally.
- Deleting a file from the UI removes both its vector entries and the uploaded source file.
- Re-uploading or re-embedding the same filename replaces old embeddings for that file.
