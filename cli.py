import typer
from main_rag import RAGPipeline
from core.llm import list_providers, detect_from_env

cli = typer.Typer(help="Modular RAG System CLI via local LLM, API providers, downloaded weights, and Chroma.")

@cli.command()
def index(source: str = typer.Argument(..., help="Path to PDF, TXT, or URL.")):
    """Index a document into the Vector Database."""
    try:
        pipeline = RAGPipeline()
        typer.echo(f"Indexing {source}...")
        result = pipeline.index_document(source)
        typer.echo(f"Success! Indexed {result['chunks_indexed']} chunks.")
    except Exception as e:
        typer.echo(f"Error during indexing: {e}", err=True)

@cli.command()
def query(
    question: str = typer.Argument(..., help="Your question."),
    session_id: str = typer.Option("default_session", help="Session ID for chat history."),
    model: str = typer.Option(None, "--model", "-m", help="LLM model (e.g. openai:gpt-4o, gemini:gemini-2.5-flash, /path/to/weights, etc.)."),
    show_context: bool = typer.Option(False, "--show-context", help="Display retrieved context."),
    stream: bool = typer.Option(False, "--stream", "-s", help="Stream the response tokens in real-time.")
):
    """Query the RAG system using configured or specified model."""
    try:
        pipeline = RAGPipeline(model=model)
        
        if stream:
            typer.echo("Thinking (streaming)...\n")
            typer.echo("Answer: ", nl=False)
            for chunk in pipeline.query_stream(session_id, question, model=model):
                typer.echo(chunk, nl=False)
            typer.echo("\n")
        else:
            typer.echo("Thinking...")
            result = pipeline.query(session_id, question, model=model)
            typer.echo(f"\nAnswer:\n{result['answer']}\n")
            typer.echo(f"(Model used: {result['model_used']})")
            
            if show_context:
                typer.echo("\n--- Retrieved Context ---")
                typer.echo(result['context_used'])
    except Exception as e:
        typer.echo(f"Error during query: {e}", err=True)

@cli.command("providers")
def show_providers():
    """List all supported LLM providers, capabilities, and configuration status."""
    providers = list_providers()
    active_env = detect_from_env()

    typer.echo("\n=== Supported LLM Providers ===")
    for p in providers:
        status = " [ACTIVE / CONFIGURED]" if (p.name in active_env or p.is_configured) else " [NOT CONFIGURED]"
        typer.echo(f"\n* {p.display_name} ({p.name}){status}")
        typer.echo(f"  Description: {p.description}")
        typer.echo(f"  Default Model: {p.default_model}")
        if p.env_vars:
            typer.echo(f"  Env Vars: {', '.join(p.env_vars)}")
        typer.echo(f"  Streaming: {p.capabilities.supports_streaming} | Max Context: {p.capabilities.max_context_window}")
    typer.echo("\nActive in current environment: " + (", ".join(active_env) if active_env else "None"))
    typer.echo("")

@cli.command("detect-env")
def show_detected_env():
    """Show providers with active API keys or endpoints detected in the environment."""
    active = detect_from_env()
    if active:
        typer.echo(f"Configured providers ready to use: {', '.join(active)}")
    else:
        typer.echo("No cloud API keys detected. Defaulting to local Ollama or Hugging Face weights.")

@cli.command()
def clear_history(session_id: str = typer.Argument("default_session", help="Session ID to clear.")):
    """Clear chat history for a given session."""
    try:
        pipeline = RAGPipeline()
        pipeline.cache.clear_chat_history(session_id)
        typer.echo(f"Cleared history for session: {session_id}")
    except Exception as e:
        typer.echo(f"Error clearing history: {e}", err=True)

@cli.command()
def clear_db():
    """Clear the vector database entirely."""
    try:
        pipeline = RAGPipeline()
        pipeline.clear_db()
        typer.echo("Database cleared successfully.")
    except Exception as e:
        typer.echo(f"Error clearing database: {e}", err=True)

@cli.command()
def deep_reset():
    """Physically delete the Chroma DB folder and start fresh."""
    try:
        pipeline = RAGPipeline()
        pipeline.deep_reset()
        typer.echo("Database directory physically deleted and re-initialized.")
    except Exception as e:
        typer.echo(f"Error during deep reset: {e}", err=True)

@cli.command()
def delete_collection(name: str = typer.Argument(..., help="Name of the collection to delete.")):
    """Delete a specific collection from the database."""
    try:
        pipeline = RAGPipeline()
        pipeline.delete_collection(name)
        typer.echo(f"Collection '{name}' deleted successfully.")
    except Exception as e:
        typer.echo(f"Error deleting collection: {e}", err=True)

if __name__ == "__main__":
    cli()
