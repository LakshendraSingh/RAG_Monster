# RAG MONSTER - (Modular Orchestrated Neural Semantic Text Embedding & Retrieval) is a Modular Production Ready RAG System (Chroma + Redis + Local Models)

A highly modular Retrieval-Augmented Generation (RAG) system running locally, designed for maximum performance, multi-turn chat context, and accurate document retrieval.

## Architecture Highlights

- **Vector Database**: Chroma via `langchain-chroma` for efficient local vector storage.
- **Caching & Chat History**: Redis for managing fast-access context windows per user session.
- **Local Embeddings**: `nomic-embed-text` running natively in Ollama (high performance, free, 768-dim).
- **Re-Ranking System**: `cross-encoder/ms-marco-MiniLM-L-6-v2` re-ranks retrieved dense chunks drastically improving retrieval utility.
- **Generation LLM**: Integrated with local models running on Ollama (defaults to `llama3:latest`).
- **Dual Interfaces**: Contains a robust FastAPI backend and a clean Typer-based CLI for testing and automation.

## Features

- **Smart Document Loaders**: Automatically handles PDF text parsing, generic `.txt`/`.md` reading, and Web crawling via `WebBaseLoader`.
- **Advanced Text Chunking**: Supports both **Semantic Chunking** (meaning-based) and **Recursive Character Splitting** (token-based).
- **Stateless & Stateful Retrieval**: Every chat is mapped to an isolated `session_id` using Redis backend.

---

## Prerequisites

1. **Python 3.9+**
2. **Chroma**: Used in-process with a local persistence directory (Default: `./chroma_db`).
3. **Redis**: You need an active Redis cache instance port (Default: 6379).
   - Redis via Docker:
     ```bash
     docker run -d -p 6379:6379 redis:latest
     ```
4. **Ollama**: An active LLM inference environment mapped locally.
   - Pull LLM: `ollama pull llama3`
   - Pull Embedding: `ollama pull nomic-embed-text`

---

## Installation & Setup

Set up an isolated python environment. Ensure dependencies are successfully fetched.

```bash
# 1. Initialize environment
python3 -m venv venv
source venv/bin/activate

# 2. Fix MacOS python build configurations (if required)
pip install -U pip setuptools wheel "cython<3" pyyaml==6.0.2

# 3. Feed the general dependencies
pip install -r requirements.txt
```

Verify the `.env` configuration file exists and correctly maps the DB instances:

```bash
cp .env.example .env
```

---

## Usage Guide

### Mode 1: Command Line Interface (CLI)

You can directly index text assets and prompt the RAG models through the terminal:

**1. Index a local file or URL**

```bash
python cli.py index "path/to/my_report.pdf"
```

**2. Query the Knowledge Base**

```bash
python cli.py query "Summarize the findings in the report?" --session-id "user1_session" --show-context
```

**3. Clear Session History**

```bash
python cli.py clear-history "user1_session"
```

**4. Deep Reset (Fix Sync Issues)**
If you encounter "Nothing found on disk" errors, use this to purge the storage and start fresh:

```bash
python cli.py deep-reset
```

**5. Clear Default Collection Data**

```bash
python cli.py clear-db
```

**6. Delete a Specific Collection**

```bash
python cli.py delete-collection "collection_name"
```

**7. Visualize Vector Database (Local Host)**

Explore your embeddings in a web-based **2D/3D semantic dashboard**:

```bash
streamlit run viz.py
```

- **2D/3D Mapping**: Toggle between fast 2D and immersive 3D scatter plots.
- **Vector Inspection**: View raw embedding previews (first 5 dimensions) for every chunk.
- **Collection Switching**: Use the sidebar to explore multiple collections.

### Configuration (`.env`)

You can customize the RAG pipeline behavior via the `.env` file:

- `EMBEDDING_MODEL_NAME`: Default `nomic-embed-text` (Ollama).
- `CHUNKING_STRATEGY`: 
    - `recursive`: Standard recursive character splitting (Default).
    - `semantic`: Smart splitting based on embedding meaning (Requires `langchain-experimental`).
- `LLM_MODEL`: The generation model in Ollama (e.g., `llama3`).

### Mode 2: FastAPI Server

Spin up the standard REST implementation backed by Uvicorn. This is useful if you are attaching Web frontends or external API services.

```bash
python app.py
```

The application mounts a Swagger testing environment containing Schema payloads at:  
**`http://localhost:8000/docs`**

#### Standard Endpoints

| Verb     | Endpoint        | Description                        | Payload Body Example                           |
| -------- | --------------- | ---------------------------------- | ---------------------------------------------- |
| `POST`   | `/index`        | Add text vectors to DB             | `{"source": "data/sales.pdf"}`                 |
| `POST`   | `/query`        | Synthesize completion from context | `{"session_id": "test_id", "question": "..."}` |
| `DELETE` | `/history/{id}` | Flush memory context matching ID   | N/A                                            |
| `DELETE` | `/db`           | Wipe the vector database           | N/A                                            |
