from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

CORPUS_DIR = Path(__file__).parent.parent / "corpus"

def load_corpus_documents(corpus_dir: Path = CORPUS_DIR) -> list[Document]:
    """Load every PDF and text file in the corpus folder as documents."""
    documents = []
    for file_path in sorted(corpus_dir.iterdir()):
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            documents.extend(PyPDFLoader(str(file_path)).load())
        elif suffix in (".txt", ".md"):
            documents.extend(TextLoader(str(file_path)).load())
    if not documents:
        raise FileNotFoundError(f"No documents loaded from {corpus_dir}. Add PDFs or .txt files there first.")
    return documents