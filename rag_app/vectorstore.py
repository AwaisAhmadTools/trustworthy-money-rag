from pathlib import Path
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
import os

PERSIST_DIR = Path(__file__).parent.parent / "chroma_db"
COLLECTION = "trustworthy_money"
EMBEDDING_MODEL = "text-embedding-3-small"

def get_embeddings() -> OpenAIEmbeddings:
    """Uses OPENAI_API_KEY from the environment (.env)."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY not set. Add it to .env and call load_dotenv() first."
        )
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)

def build_index(chunks: list[Document], persist_dir: Path = PERSIST_DIR) -> Chroma:
    """Embed the chunks and persist the index. This is the ONLY step that costs API calls.
    Each record stores: vector(compute by embedding model), text(page_content), metadata(source,title, page..), ID
    """
    return Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=COLLECTION,
        persist_directory=str(persist_dir),
    )

def load_index(persist_dir: Path = PERSIST_DIR) -> Chroma:
    """Load the persisted index from disk.

    No documents are re-embedded (that was build_index's job).
    Note: each subsequent search embeds the query — one API call per search.
    """
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=str(persist_dir),
    )

def search(query: str, k: int = 3, persist_dir: Path = PERSIST_DIR) -> list:
    """Top-k chunks with relevance scores (0–1, higher = closer)."""
    store = load_index(persist_dir)
    return store.similarity_search_with_relevance_scores(query, k=k)