import os
from dotenv import load_dotenv

load_dotenv()

# Vector Database (Chroma)
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")

# Redis
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Generation
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3:latest")

# Embedding
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "nomic-embed-text")

# Server Config
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# Chunking
CHUNKING_STRATEGY = os.getenv("CHUNKING_STRATEGY", "recursive") # Options: recursive, semantic
