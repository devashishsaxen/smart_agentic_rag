# 🤖 Agentic RAG — Claude Code Prompt

> Paste this entire prompt into Claude Code to scaffold a full Agentic RAG system in one shot.

---

## PROMPT START

You are an expert Python engineer. Build me a complete, working **Agentic RAG (Retrieval-Augmented Generation)** system from scratch in a single project folder called `agentic_rag/`.

---

### 🎯 GOAL

Build a system where:
1. A user can **ingest documents** (PDFs and .txt files) into a local vector database
2. An **AI agent** decides on its own whether to retrieve context from the vector DB or answer directly
3. The agent uses **tool calling** to invoke the retriever when needed
4. The user can **chat** with the agent via a clean terminal interface

---

### 📁 PROJECT STRUCTURE

Create exactly this structure:

```
agentic_rag/
├── main.py                  # Entry point — chat loop
├── ingest.py                # Document ingestion pipeline
├── retriever.py             # Vector DB retrieval tool
├── agent.py                 # Agentic loop with tool use
├── config.py                # All config and constants
├── requirements.txt         # All dependencies pinned
├── .env.example             # Template for API keys
├── data/
│   └── sample.txt           # A sample document for testing
└── README.md                # Setup and usage instructions
```

---

### 🛠️ TECH STACK (use exactly these)

- **LLM:** Anthropic Claude (`claude-sonnet-4-20250514`) via `anthropic` Python SDK
- **Vector DB:** `chromadb` (local, persistent, no external server)
- **Embeddings:** `sentence-transformers` with model `all-MiniLM-L6-v2` (free, local)
- **PDF parsing:** `pypdf`
- **Text splitting:** Manual chunking (no LangChain dependency)
- **Env management:** `python-dotenv`

Do NOT use LangChain, LlamaIndex, or any heavy framework. Keep it raw and minimal.

---

### ⚙️ config.py

```python
# All constants in one place
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "rag_documents"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500          # characters per chunk
CHUNK_OVERLAP = 50        # overlap between chunks
TOP_K_RESULTS = 3         # number of chunks to retrieve
CLAUDE_MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 1024
```

---

### 📥 ingest.py — Document Ingestion

Build a script that:
1. Accepts a file path as a CLI argument: `python ingest.py ./data/my_doc.pdf`
2. Supports `.pdf` and `.txt` files
3. Splits text into overlapping chunks (use `CHUNK_SIZE` and `CHUNK_OVERLAP` from config)
4. Generates embeddings using `sentence-transformers`
5. Stores chunks + embeddings + metadata (filename, chunk index) into ChromaDB
6. Prints progress: how many chunks were created and stored
7. Is idempotent — re-running with the same file should overwrite, not duplicate

---

### 🔍 retriever.py — Retrieval Tool

Build a `retrieve(query: str) -> str` function that:
1. Embeds the query using the same `sentence-transformers` model
2. Queries ChromaDB for top-K most similar chunks
3. Returns a formatted string like:

```
--- Retrieved Context ---
[Source: sample.txt | Chunk 3]
...chunk text here...

[Source: sample.txt | Chunk 7]
...chunk text here...
-------------------------
```

4. If no results found, returns: `"No relevant context found in the knowledge base."`

Also expose the tool definition as a Python dict for Claude's tool_use API:

```python
RETRIEVER_TOOL = {
    "name": "retrieve_context",
    "description": "Search the knowledge base for relevant information to answer the user's question. Use this whenever the question might be answered by ingested documents.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to find relevant context"
            }
        },
        "required": ["query"]
    }
}
```

---

### 🤖 agent.py — Agentic Loop

Build an `Agent` class with:

**`__init__`:**
- Initializes conversation history as an empty list
- Loads system prompt

**System prompt must say:**
> "You are a helpful assistant with access to a knowledge base. When a user asks a question, decide if you need to search the knowledge base using the `retrieve_context` tool. If the question is general knowledge (greetings, math, etc.), answer directly. If the question might be about documents the user has ingested, always search first."

**`chat(user_message: str) -> str` method:**
1. Append user message to history
2. Call Claude API with:
   - Full conversation history
   - The `RETRIEVER_TOOL` definition
   - `tool_choice = {"type": "auto"}`
3. If Claude returns a `tool_use` block:
   - Extract the query from tool input
   - Call `retrieve(query)` from `retriever.py`
   - Append the tool result back to history as a `tool_result` message
   - Call Claude API again with updated history
4. Extract final text response
5. Append assistant response to history
6. Return the final text response

Handle multi-turn correctly — history must persist across calls.

---

### 💬 main.py — Chat Interface

Build a terminal chat loop that:
1. On startup, prints:
```
🤖 Agentic RAG Assistant
Type 'exit' to quit | Type 'ingest <filepath>' to add documents
----------------------------------------
```
2. Accepts user input in a loop
3. If input starts with `ingest `, calls the ingestion pipeline inline
4. Otherwise, passes the message to `Agent.chat()` and prints the response
5. Shows a simple `[thinking...]` indicator while waiting
6. Gracefully handles `KeyboardInterrupt`

---

### 📦 requirements.txt

Pin these exact packages:
```
anthropic>=0.40.0
chromadb>=0.5.0
sentence-transformers>=3.0.0
pypdf>=4.0.0
python-dotenv>=1.0.0
```

---

### 📄 data/sample.txt

Create a sample document with ~300 words of content about a **fictional company called "NovaTech"** — its products, founding year (2019), CEO name (Aria Chen), and main product (an AI scheduling assistant called "FlowAI"). This will be used to test retrieval.

---

### 📖 README.md

Include:
1. Prerequisites (Python 3.10+, pip)
2. Installation steps (clone, venv, pip install, .env setup)
3. How to ingest a document
4. How to start chatting
5. Example conversation showing the agent using the retriever tool
6. Brief explanation of how the agentic loop works

---

### ✅ ACCEPTANCE CRITERIA

Before finishing, verify:
- [ ] `pip install -r requirements.txt` runs without errors
- [ ] `python ingest.py data/sample.txt` successfully stores chunks in ChromaDB
- [ ] `python main.py` starts a chat session
- [ ] Asking "Who is the CEO of NovaTech?" triggers a `retrieve_context` tool call and returns the correct answer
- [ ] Asking "What is 2 + 2?" does NOT trigger retrieval and is answered directly
- [ ] Re-ingesting the same file does not create duplicate chunks

---

### ⚠️ IMPORTANT RULES

1. **No placeholder code** — every function must be fully implemented
2. **No LangChain / LlamaIndex** — pure Python + the listed libraries only
3. **All errors must be caught** with helpful messages (e.g., "File not found", "API key missing")
4. **`.env.example`** must contain: `ANTHROPIC_API_KEY=your_key_here`
5. **ChromaDB** must be configured to persist to disk (not in-memory)
6. The agent must maintain **multi-turn conversation history** correctly
7. Comments in code should explain the "why", not the "what"

---

### 🏁 FINAL STEP

After generating all files, run:
```bash
python ingest.py data/sample.txt
```
and show me the output to confirm ingestion works.

## PROMPT END
