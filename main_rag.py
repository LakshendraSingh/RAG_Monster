import os
import langchain
# Monkeypatch for langchain.verbose if missing in newer versions
if not hasattr(langchain, "verbose"):
    langchain.verbose = False
from core.document_loader import DocumentLoader
from core.chunker import DocumentChunker
from core.vdb import VectorDB
from core.retriever import RAGRetriever
from core.cache import CacheManager
from config import OLLAMA_BASE_URL, LLM_MODEL
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

class RAGPipeline:
    """Orchestrates the components to form the full RAG system."""
    
    def __init__(self, use_reranker: bool = True):
        self.vdb = VectorDB()
        self.retriever = RAGRetriever(use_reranker=use_reranker)
        self.cache = CacheManager()
        self.llm = Ollama(base_url=OLLAMA_BASE_URL, model=LLM_MODEL)
        
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question", "chat_history"],
            template="""You are a highly capable AI assistant powered by a Modular RAG system.
Use the following pieces of relevant context and previous chat history to answer the question.
If the answer cannot be found in the context, explicitly mention that, but try your best to synthesize a useful response.

--- Chat History ---
{chat_history}

--- Context ---
{context}

--- Question ---
{question}

Answer:"""
        )

    def index_document(self, file_path_or_url: str):
        """Loads, chunks, and indexes a document into the Vector Database."""
        from config import CHUNKING_STRATEGY
        from core.embedder import Embedder
        
        print(f"Loading document from: {file_path_or_url}")
        docs = DocumentLoader.load_any(file_path_or_url)
        
        if CHUNKING_STRATEGY == "semantic":
            print(f"Semantic Chunking {len(docs)} documents...")
            embeddings = Embedder.get_embeddings()
            chunked_docs = DocumentChunker.chunk_documents_semantic(docs, embeddings)
        else:
            print(f"Recursive Chunking {len(docs)} documents (default)...")
            chunked_docs = DocumentChunker.chunk_documents_recursive(docs)
        
        print(f"Indexing {len(chunked_docs)} chunks into Chroma...")
        self.vdb.add_documents(chunked_docs)
        
        return {"status": "success", "chunks_indexed": len(chunked_docs), "source": file_path_or_url}

    def query(self, session_id: str, question: str):
        """Retrieves context, formats prompt, and generates answer incorporating history."""
        # 1. Get Chat History
        history = self.cache.get_chat_history(session_id)
        history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history[-5:]]) # Last 5 turns
        
        # 2. Retrieve Context
        print("Retrieving context...")
        context = self.retriever.retrieve_and_format(question)
        
        # 3. Generate Answer
        print("Generating response...")
        prompt = self.prompt_template.format(
            context=context,
            question=question,
            chat_history=history_str
        )
        answer = self.llm.invoke(prompt)
        
        # 4. Update Cache
        self.cache.add_chat_message(session_id, "user", question)
        self.cache.add_chat_message(session_id, "assistant", answer)
        
        return {"answer": answer, "context_used": context}

    def clear_db(self):
        """Clear the vector database."""
        self.vdb.clear_database()
        return {"status": "success"}

    def deep_reset(self):
        """Physically delete the Chroma directory for a full reset."""
        import shutil
        from config import CHROMA_PATH
        if os.path.exists(CHROMA_PATH):
            shutil.rmtree(CHROMA_PATH)
        # Re-initialize
        self.vdb = VectorDB()
        return {"status": "success"}

    def delete_collection(self, name: str):
        """Delete a specific collection."""
        self.vdb.delete_collection(name)
        return {"status": "success"}
