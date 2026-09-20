import tiktoken
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Token-based sizing: exact match to what the model consumes (limits + cost).
# o200k_base = the GPT-4o family encoding. 500 tokens ≈ 2000 chars.
ENCODING = tiktoken.encoding_for_model("gpt-4o")

def token_length(text: str) -> int:
    """Count tokens using the model's real encoding."""
    return len(ENCODING.encode(text))

def chunk_documents(documents: list[Document],
                    chunk_size: int = 500,      # tokens: 1 token ~ 4 chars
                    chunk_overlap: int = 75) -> list[Document]:   # ~15% overlap
    """Split documents into token-sized chunks retrievable by the vector store."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", ".\n", ". ", "\n", " ", ""],
        keep_separator="end",                              # ← punctuation stays with ITS chunk 
        length_function=token_length,   # ← the switch from chars to tokens
    )
    return splitter.split_documents(documents)