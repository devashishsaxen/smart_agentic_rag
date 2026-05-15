import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

from agent import Agent
from config import MEMORY_FILE, UPLOAD_DIR
from ingest import (
    delete_document,
    ensure_upload_dir,
    ingest_document,
    list_documents,
    resolve_uploaded_file,
)


load_dotenv()
app = FastAPI(title="Agentic RAG Web App")
_agent: Agent | None = None
ALLOWED_EXTENSIONS = {".pdf", ".txt"}


def get_agent() -> Agent:
    global _agent
    if _agent is None:
        _agent = Agent()
    return _agent


def validate_filename(filename: str) -> str:
    cleaned = Path(filename).name.strip()
    if not cleaned:
        raise HTTPException(status_code=400, detail="Filename is required.")

    suffix = Path(cleaned).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only .pdf and .txt files are supported.")

    return cleaned


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Agentic RAG Console</title>
  <style>
    :root {
      --bg: #f4efe6;
      --panel: rgba(255, 252, 247, 0.88);
      --panel-strong: #fffaf2;
      --ink: #1d1a16;
      --muted: #6f665b;
      --accent: #c25b2f;
      --accent-dark: #8b3f1f;
      --line: rgba(29, 26, 22, 0.12);
      --success: #2f7d4a;
      --danger: #b53a2d;
      --shadow: 0 18px 48px rgba(56, 40, 23, 0.14);
    }

    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(194, 91, 47, 0.16), transparent 28%),
        radial-gradient(circle at top right, rgba(81, 111, 94, 0.12), transparent 24%),
        linear-gradient(160deg, #f8f1e7 0%, #efe5d6 48%, #e7dccd 100%);
      min-height: 100vh;
    }

    .shell {
      width: min(1180px, calc(100% - 32px));
      margin: 24px auto;
      display: grid;
      grid-template-columns: 380px minmax(0, 1fr);
      gap: 20px;
    }

    .panel {
      background: var(--panel);
      backdrop-filter: blur(18px);
      border: 1px solid rgba(255, 255, 255, 0.6);
      border-radius: 24px;
      box-shadow: var(--shadow);
    }

    .sidebar {
      padding: 22px;
      display: flex;
      flex-direction: column;
      gap: 18px;
    }

    .hero h1 {
      margin: 0;
      font-size: 2rem;
      line-height: 1;
      letter-spacing: -0.03em;
    }

    .hero p, .muted { color: var(--muted); }

    .card {
      background: var(--panel-strong);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 16px;
    }

    h2, h3 {
      margin: 0 0 10px;
      font-size: 1rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }

    .upload-row {
      display: flex;
      flex-direction: column;
      gap: 10px;
    }

    input[type="file"] {
      width: 100%;
      padding: 10px;
      border-radius: 12px;
      border: 1px dashed var(--line);
      background: #fff;
    }

    button {
      border: 0;
      border-radius: 999px;
      padding: 11px 16px;
      background: var(--accent);
      color: #fff;
      cursor: pointer;
      font-weight: 700;
      transition: transform 160ms ease, background 160ms ease;
    }

    button:hover { transform: translateY(-1px); background: var(--accent-dark); }
    button.secondary { background: #ded3c3; color: var(--ink); }
    button.danger { background: var(--danger); }
    button:disabled { opacity: 0.6; cursor: wait; transform: none; }

    .status {
      min-height: 20px;
      font-size: 0.95rem;
      color: var(--muted);
    }

    .status.error { color: var(--danger); }
    .status.success { color: var(--success); }

    .doc-list {
      display: flex;
      flex-direction: column;
      gap: 12px;
      max-height: 52vh;
      overflow: auto;
      padding-right: 4px;
    }

    .doc {
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 14px;
      background: #fffdf9;
    }

    .doc-head {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: start;
    }

    .doc-name {
      margin: 0;
      font-size: 1rem;
      word-break: break-word;
    }

    .badge {
      display: inline-block;
      font-size: 0.75rem;
      padding: 4px 8px;
      border-radius: 999px;
      background: #f0e1d2;
      color: var(--accent-dark);
      margin-top: 8px;
    }

    .doc-meta {
      margin: 10px 0 0;
      color: var(--muted);
      font-size: 0.92rem;
    }

    .doc-actions {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-top: 12px;
    }

    .chat-panel {
      display: flex;
      flex-direction: column;
      min-height: calc(100vh - 48px);
      overflow: hidden;
    }

    .chat-top {
      padding: 22px 24px 10px;
      border-bottom: 1px solid var(--line);
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: center;
    }

    .chat-top h2 {
      margin-bottom: 4px;
      font-size: 1.1rem;
    }

    .chat-stream {
      padding: 22px 24px;
      display: flex;
      flex-direction: column;
      gap: 14px;
      overflow: auto;
      flex: 1;
    }

    .bubble {
      max-width: 85%;
      padding: 14px 16px;
      border-radius: 18px;
      white-space: pre-wrap;
      line-height: 1.5;
      animation: rise 220ms ease;
    }

    .user {
      align-self: flex-end;
      background: #e7c5b1;
    }

    .assistant {
      align-self: flex-start;
      background: #fff;
      border: 1px solid var(--line);
    }

    @keyframes rise {
      from { opacity: 0; transform: translateY(10px); }
      to { opacity: 1; transform: translateY(0); }
    }

    .composer {
      padding: 18px 24px 24px;
      border-top: 1px solid var(--line);
      background: linear-gradient(180deg, rgba(255,250,242,0.65), rgba(255,250,242,0.95));
    }

    .composer form {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 12px;
    }

    textarea {
      width: 100%;
      resize: vertical;
      min-height: 76px;
      max-height: 180px;
      border-radius: 18px;
      border: 1px solid var(--line);
      padding: 14px;
      font: inherit;
      background: #fff;
      color: var(--ink);
    }

    @media (max-width: 960px) {
      .shell { grid-template-columns: 1fr; }
      .chat-panel { min-height: 72vh; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <aside class="panel sidebar">
      <div class="hero">
        <h1>RAG Control Room</h1>
        <p>Upload documents, manage embeddings, and chat against the stored knowledge base.</p>
      </div>

      <section class="card">
        <h2>Upload</h2>
        <div class="upload-row">
          <input id="fileInput" type="file" accept=".pdf,.txt" />
          <button id="uploadBtn">Upload and Embed</button>
          <div id="uploadStatus" class="status"></div>
        </div>
      </section>

      <section class="card">
        <h2>Stored Files</h2>
        <div id="docList" class="doc-list"></div>
      </section>
    </aside>

    <main class="panel chat-panel">
      <div class="chat-top">
        <div>
          <h2>Chatbot</h2>
          <div class="muted">Ask questions over your uploaded documents or use it like a normal assistant.</div>
        </div>
        <button id="clearMemoryBtn" class="secondary">Clear Chat Memory</button>
      </div>
      <div id="chatStream" class="chat-stream">
        <div class="bubble assistant">The assistant is ready. Upload a file or start chatting.</div>
      </div>
      <div class="composer">
        <form id="chatForm">
          <textarea id="messageInput" placeholder="Ask something about your files..."></textarea>
          <button id="sendBtn" type="submit">Send</button>
        </form>
        <div id="chatStatus" class="status"></div>
      </div>
    </main>
  </div>

  <script>
    const docList = document.getElementById("docList");
    const uploadStatus = document.getElementById("uploadStatus");
    const chatStatus = document.getElementById("chatStatus");
    const chatStream = document.getElementById("chatStream");
    const fileInput = document.getElementById("fileInput");
    const uploadBtn = document.getElementById("uploadBtn");
    const clearMemoryBtn = document.getElementById("clearMemoryBtn");
    const chatForm = document.getElementById("chatForm");
    const messageInput = document.getElementById("messageInput");
    const sendBtn = document.getElementById("sendBtn");

    function setStatus(target, message, kind = "") {
      target.textContent = message || "";
      target.className = kind ? `status ${kind}` : "status";
    }

    function addBubble(text, role) {
      const bubble = document.createElement("div");
      bubble.className = `bubble ${role}`;
      bubble.textContent = text;
      chatStream.appendChild(bubble);
      chatStream.scrollTop = chatStream.scrollHeight;
    }

    function formatUploadedAt(value) {
      if (!value) return "Not uploaded through the web app";
      return new Date(value).toLocaleString();
    }

    function renderDocuments(items) {
      docList.innerHTML = "";
      if (!items.length) {
        docList.innerHTML = '<div class="muted">No stored files yet.</div>';
        return;
      }

      for (const item of items) {
        const card = document.createElement("article");
        card.className = "doc";
        card.innerHTML = `
          <div class="doc-head">
            <div>
              <h3 class="doc-name">${item.filename}</h3>
              <span class="badge">${item.embedded ? "Embedded" : "Uploaded only"}</span>
            </div>
          </div>
          <div class="doc-meta">Type: ${item.file_type || "unknown"} | Chunks: ${item.chunk_count} | Pages: ${item.page_count}</div>
          <div class="doc-meta">Uploaded: ${formatUploadedAt(item.uploaded_at)}</div>
          <div class="doc-actions">
            <button class="secondary" data-action="reembed" data-name="${item.filename}">Re-embed</button>
            <button class="danger" data-action="delete" data-name="${item.filename}">Delete</button>
          </div>
        `;
        docList.appendChild(card);
      }
    }

    async function loadDocuments() {
      const response = await fetch("/api/documents");
      const payload = await response.json();
      renderDocuments(payload.documents || []);
    }

    async function uploadDocument() {
      const file = fileInput.files[0];
      if (!file) {
        setStatus(uploadStatus, "Choose a .pdf or .txt file first.", "error");
        return;
      }

      uploadBtn.disabled = true;
      setStatus(uploadStatus, "Uploading and generating embeddings...");
      const formData = new FormData();
      formData.append("file", file);

      try {
        const response = await fetch("/api/documents", { method: "POST", body: formData });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.detail || "Upload failed.");
        setStatus(uploadStatus, `Stored ${payload.filename} with ${payload.chunk_count} chunks.`, "success");
        fileInput.value = "";
        await loadDocuments();
      } catch (error) {
        setStatus(uploadStatus, error.message, "error");
      } finally {
        uploadBtn.disabled = false;
      }
    }

    async function reembedDocument(filename) {
      setStatus(uploadStatus, `Re-embedding ${filename}...`);
      try {
        const response = await fetch(`/api/documents/${encodeURIComponent(filename)}/reembed`, { method: "POST" });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.detail || "Re-embed failed.");
        setStatus(uploadStatus, `Re-embedded ${payload.filename} with ${payload.chunk_count} chunks.`, "success");
        await loadDocuments();
      } catch (error) {
        setStatus(uploadStatus, error.message, "error");
      }
    }

    async function deleteDocument(filename) {
      setStatus(uploadStatus, `Deleting ${filename}...`);
      try {
        const response = await fetch(`/api/documents/${encodeURIComponent(filename)}`, { method: "DELETE" });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.detail || "Delete failed.");
        setStatus(uploadStatus, payload.message, "success");
        await loadDocuments();
      } catch (error) {
        setStatus(uploadStatus, error.message, "error");
      }
    }

    async function sendMessage(event) {
      event.preventDefault();
      const message = messageInput.value.trim();
      if (!message) return;

      addBubble(message, "user");
      messageInput.value = "";
      sendBtn.disabled = true;
      setStatus(chatStatus, "Assistant is thinking...");

      try {
        const response = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message })
        });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.detail || "Chat failed.");
        addBubble(payload.response, "assistant");
        setStatus(chatStatus, "", "");
      } catch (error) {
        addBubble(error.message, "assistant");
        setStatus(chatStatus, error.message, "error");
      } finally {
        sendBtn.disabled = false;
      }
    }

    async function clearMemory() {
      clearMemoryBtn.disabled = true;
      try {
        const response = await fetch("/api/chat/clear", { method: "POST" });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.detail || "Could not clear memory.");
        addBubble("Chat memory cleared.", "assistant");
      } catch (error) {
        setStatus(chatStatus, error.message, "error");
      } finally {
        clearMemoryBtn.disabled = false;
      }
    }

    uploadBtn.addEventListener("click", uploadDocument);
    clearMemoryBtn.addEventListener("click", clearMemory);
    chatForm.addEventListener("submit", sendMessage);
    docList.addEventListener("click", async (event) => {
      const button = event.target.closest("button[data-action]");
      if (!button) return;
      const action = button.dataset.action;
      const filename = button.dataset.name;
      if (action === "reembed") await reembedDocument(filename);
      if (action === "delete") await deleteDocument(filename);
    });

    loadDocuments();
  </script>
</body>
</html>"""


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "upload_dir": str(Path(UPLOAD_DIR).resolve())}


@app.get("/api/documents")
def get_documents() -> dict:
    return {"documents": list_documents()}


@app.post("/api/documents")
async def upload_document(file: UploadFile = File(...)) -> dict:
    filename = validate_filename(file.filename or "")
    upload_dir = ensure_upload_dir()
    target = upload_dir / filename

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    target.write_bytes(data)

    try:
        chunk_count = ingest_document(str(target))
    except Exception as exc:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Ingestion failed: {exc}") from exc

    return {"filename": filename, "chunk_count": chunk_count}


@app.post("/api/documents/{filename}/reembed")
def reembed_document(filename: str) -> dict:
    cleaned = validate_filename(filename)
    try:
        path = resolve_uploaded_file(cleaned)
        chunk_count = ingest_document(str(path))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Re-embedding failed: {exc}") from exc

    return {"filename": cleaned, "chunk_count": chunk_count}


@app.delete("/api/documents/{filename}")
def remove_document(filename: str) -> dict:
    cleaned = validate_filename(filename)
    removed_chunks = delete_document(cleaned)
    file_path = ensure_upload_dir() / cleaned
    file_removed = False
    if file_path.exists():
        file_path.unlink()
        file_removed = True

    return {
        "message": (
            f"Removed {cleaned}. Deleted {removed_chunks} embedded chunks"
            + (" and the uploaded file." if file_removed else ".")
        )
    }


@app.post("/api/chat")
def chat(payload: dict) -> dict:
    message = (payload.get("message") or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message is required.")

    if not os.getenv("NVIDIA_API_KEY"):
        raise HTTPException(
            status_code=500,
            detail="NVIDIA_API_KEY is missing. Add it to your environment or .env file.",
        )

    try:
        response = get_agent().chat(message)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"response": response}


@app.post("/api/chat/clear")
def clear_chat() -> dict:
    try:
        if _agent is not None:
            _agent.clear_memory()
        else:
            memory_file = Path(MEMORY_FILE)
            if memory_file.exists():
                memory_file.unlink()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"message": "Chat memory cleared."}
