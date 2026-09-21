from dotenv import load_dotenv
load_dotenv() 
from rag_app.loader import load_corpus_documents
from rag_app.chunker import chunk_documents
from rag_app.vectorstore import build_index, load_index, search, PERSIST_DIR
from rag_app.config import EXPECTED_CHUNKS, TEST_QUERY

if not PERSIST_DIR.exists():
    chunks = chunk_documents(load_corpus_documents())
    store = build_index(chunks) # costs ~1p — the only paid step
else:
    store = load_index()

count = store._collection.count()
assert count == EXPECTED_CHUNKS, (
    f"expected {EXPECTED_CHUNKS} records, got {count} — duplicate build? delete chroma_db/ and rebuild"
)
print(f"records: {count} ✓")                    

results = search(TEST_QUERY, k=3)
for doc, score in results:
    print(f"{score:.3f} | {doc.metadata.get('title')} | page {doc.metadata.get('page')}")
    print(doc.page_content[:200], "\n---")