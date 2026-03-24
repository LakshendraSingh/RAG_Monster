import typer
from main_rag import RAGPipeline

cli = typer.Typer(help="Modular RAG System CLI via local LLM and Chroma.")

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
    show_context: bool = typer.Option(False, "--show-context", help="Display retrieved context.")
):
    """Query the RAG system."""
    try:
        pipeline = RAGPipeline()
        typer.echo("Thinking...")
        result = pipeline.query(session_id, question)
        
        typer.echo(f"\nAnswer:\n{result['answer']}\n")
        
        if show_context:
            typer.echo("\n--- Retrieved Context ---")
            typer.echo(result['context_used'])
    except Exception as e:
        typer.echo(f"Error during query: {e}", err=True)

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
