# RAG MONSTER - (Modular Orchestrated Neural Semantic Text Embedding & Retrieval)

A modular, production-ready Retrieval-Augmented Generation (RAG) system with Chroma vector storage, Redis caching (with in-memory fallback), and a universal LLM engine supporting **cloud API-based LLMs** and **Hugging Face / GGUF downloaded weights** (inspired by the LucyCode multi-provider architecture).

---

## Architecture Highlights

- **Universal Multi-Provider LLM Engine**: Seamlessly switch between cloud APIs (OpenAI, Anthropic, Gemini, Groq, DeepSeek, Mistral, OpenRouter, Together, Fireworks), direct Hugging Face downloaded weights (via `transformers` + `torch` with Apple Silicon MPS / CUDA), direct GGUF files (via `llama-cpp-python`), or local servers (Ollama, llama.cpp, vLLM).
- **Flexible Embeddings**: Pluggable embedding models supporting Ollama (`nomic-embed-text`), HuggingFace / Sentence-Transformers (`sentence-transformers/all-MiniLM-L6-v2`), and OpenAI (`text-embedding-3-small`).
- **Resilient Caching & Session History**: Redis backend for fast context window management, with automatic transparent **in-memory fallback** if Redis is not running.
- **Vector Database**: Chroma via `langchain-chroma` for efficient local vector storage.
- **Re-Ranking System**: `cross-encoder/ms-marco-MiniLM-L-6-v2` re-ranks retrieved dense chunks drastically improving retrieval utility.
- **Dual Interfaces**: Robust FastAPI backend with Server-Sent Events (SSE) streaming and a rich Typer-based CLI for testing and automation.

---

## Supported LLM Providers & Models

RAG_MONSTER detects the appropriate provider automatically using model prefixes or file paths:

### 1. Cloud API Providers
| Provider | Prefix / Example Model | Required Environment Variable |
|---|---|---|
| **OpenAI** | `openai:gpt-4o`, `gpt-4o-mini`, `o1`, `o3-mini` | `OPENAI_API_KEY` |
| **Anthropic** | `anthropic:claude-3-5-sonnet-20241022`, `claude-3-5-haiku` | `ANTHROPIC_API_KEY` |
| **Google Gemini** | `gemini:gemini-flash-latest`, `gemini-pro-latest` | `GEMINI_API_KEY` or `GOOGLE_API_KEY` |
| **Groq Cloud** | `groq:llama-3.3-70b-versatile` | `GROQ_API_KEY` |
| **DeepSeek** | `deepseek:deepseek-chat`, `deepseek:deepseek-reasoner` | `DEEPSEEK_API_KEY` |
| **Mistral AI** | `mistral:mistral-large-latest`, `codestral-latest` | `MISTRAL_API_KEY` |
| **OpenRouter** | `openrouter:anthropic/claude-3.5-sonnet` | `OPENROUTER_API_KEY` |
| **Together AI** | `together:meta-llama/Llama-3.3-70B-Instruct-Turbo` | `TOGETHER_API_KEY` |
| **Fireworks AI** | `fireworks:accounts/fireworks/models/llama-v3p3-70b-instruct` | `FIREWORKS_API_KEY` |

### 2. Direct Hugging Face & Local Weights
Load downloaded weights directly into memory without requiring an external server process:

- **Hugging Face Model Directory**:
  Point to any directory containing `config.json` and `.safetensors` (or download via `git lfs` / `huggingface-cli`):
  ```bash
  python cli.py query "Explain quantum computing" --model "/path/to/downloaded/Meta-Llama-3-8B-Instruct"
  ```
  *(Automatically detects Apple Silicon MPS, CUDA, or CPU fallback, and injects correct chat templates).*

- **GGUF Weights Files**:
  Point to any `.gguf` file:
  ```bash
  python cli.py query "Explain quantum computing" --model "weights:/path/to/model.gguf"
  ```

- **Global In-Memory Cache**: Weights are retained in memory across queries to avoid reloading latency.

### 3. Local Inference Servers
- **Ollama**: `ollama:llama3:latest` (or `llama3:latest`) via `OLLAMA_BASE_URL` (default `http://localhost:11434`).
- **llama.cpp**: `llamacpp:local-model` via `LLAMACPP_HOST` (default `http://localhost:8080`).
- **OpenAI-Compatible Servers** (vLLM, LMStudio, LocalAI): `openai_compat:my-model` with `OPENAI_BASE_URL`.

---

## Installation & Setup

1. **Clone repository and setup Python virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. *(Optional)* **Install dependencies for local weights execution**:
   If you plan to run downloaded Hugging Face safetensors or GGUF weights locally:
   ```bash
   pip install torch transformers accelerate
   # For GGUF support:
   pip install llama-cpp-python
   ```

3. **Configure environment**:
   ```bash
   cp .env.example .env
   ```
   Add any API keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, etc.) or configure `LLM_MODEL`.
   > **Note**: Your `.env` file is excluded from Git tracking via `.gitignore` to protect your credentials.

4. **Background services (Optional)**:
   - Redis: Optional (the system automatically falls back to an in-memory cache if Redis is not running).
     ```bash
     docker run -d -p 6379:6379 redis:latest
     ```
   - Ollama (if using local embeddings or local models):
     ```bash
     ollama pull nomic-embed-text
     ollama pull llama3
     ```

---

## Usage Guide

### Mode 1: Command Line Interface (CLI)

**1. Inspect Providers and Active Environment**:
```bash
# List all supported providers and their configuration status
python cli.py providers

# Show active providers detected from environment API keys
python cli.py detect-env
```

**2. Index Documents**:
```bash
python cli.py index "path/to/my_report.pdf"
```

**3. Query using default model**:
```bash
python cli.py query "Summarize the report" --show-context
```

**4. Query with specific API model on-the-fly**:
```bash
# Google Gemini Flash
python cli.py query "Summarize key findings" --model "gemini:gemini-flash-latest"

# OpenAI GPT-4o
python cli.py query "Summarize key findings" --model "openai:gpt-4o"

# Anthropic Claude 3.5 Sonnet
python cli.py query "Summarize key findings" --model "anthropic:claude-3-5-sonnet-20241022"

# Groq LPU
python cli.py query "Summarize key findings" --model "groq:llama-3.3-70b-versatile"
```

**5. Query with downloaded Hugging Face weights or GGUF**:
```bash
# Hugging Face directory
python cli.py query "Explain section 2" --model "./models/Llama-3-8B-Instruct"

# GGUF file
python cli.py query "Explain section 2" --model "weights:./models/mistral-7b.gguf"
```

**6. Stream tokens in real time**:
```bash
python cli.py query "Draft an executive summary" --model "gemini:gemini-flash-latest" --stream
```

**7. Database and Session Management**:
```bash
# Clear chat history for session
python cli.py clear-history "default_session"

# Clear vector database
python cli.py clear-db

# Deep reset (physically purges and re-initializes Chroma directory)
python cli.py deep-reset
```

---

### Mode 2: FastAPI Server

Start the REST API server:
```bash
python app.py
```
Interactive Swagger API docs are available at **`http://localhost:8000/docs`**.

#### API Endpoints

| Method | Endpoint | Description | Payload Body Example |
|---|---|---|---|
| `POST` | `/index` | Add text vectors to Chroma DB | `{"source": "data/sales.pdf"}` |
| `POST` | `/query` | Query RAG with session memory | `{"session_id": "user1", "question": "...", "model": "gemini:gemini-flash-latest"}` |
| `POST` | `/query/stream` | Stream tokens using Server-Sent Events | `{"session_id": "user1", "question": "...", "model": "gemini:gemini-flash-latest"}` |
| `GET` | `/providers` | List all supported providers & status | N/A |
| `DELETE` | `/history/{session_id}` | Flush chat history for session | N/A |
| `DELETE` | `/db` | Clear vector database | N/A |

---

### Mode 3: Semantic Visualizer

Explore stored embeddings in an interactive 2D/3D semantic map:
```bash
streamlit run viz.py
```
