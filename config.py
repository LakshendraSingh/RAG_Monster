import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # Lightweight fallback parser if python-dotenv is not installed
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip()
                    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                        v = v[1:-1]
                    if k not in os.environ:
                        os.environ[k] = v

# Vector Database (Chroma)
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")

# Redis
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Generation / LLM Model
# Examples:
#   API models: "openai:gpt-4o", "gemini:gemini-2.5-flash", "anthropic:claude-3-5-sonnet-20241022",
#               "groq:llama-3.3-70b-versatile", "deepseek:deepseek-chat", "mistral:mistral-large-latest"
#   Hugging Face weights directory: "/path/to/downloaded-hf-weights", "./models/my-hf-model"
#   GGUF files: "weights:/path/to/model.gguf", "./models/model.gguf"
#   Local servers: "ollama:llama3", "llama3:latest", "llamacpp:local-model"
LLM_MODEL = os.getenv("LLM_MODEL", "llama3:latest")

# API Keys & Endpoints
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_BASE_URL = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", ""))

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "")
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY", "")

# Local Inference / Weights Config
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", os.getenv("OLLAMA_HOST", "http://localhost:11434"))
LLAMACPP_HOST = os.getenv("LLAMACPP_HOST", "http://localhost:8080")
WEIGHTS_DEVICE = os.getenv("WEIGHTS_DEVICE", "auto") # auto, mps, cuda, cpu

# Embedding
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "nomic-embed-text")

# Server Config
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# Chunking
CHUNKING_STRATEGY = os.getenv("CHUNKING_STRATEGY", "recursive") # Options: recursive, semantic
