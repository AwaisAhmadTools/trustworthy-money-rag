
from dotenv import load_dotenv
load_dotenv() 
from rag_app.loader import load_corpus_documents
from rag_app.chunker import chunk_documents
from rag_app.vectorstore import build_index, load_index, search

chunks = chunk_documents(load_corpus_documents())
store = build_index(chunks)                    # costs ~1p — the only paid step

results = search("What is murabaha and how does it differ from a conventional loan?", k=3)
for doc, score in results:
    print(f"{score:.3f} | {doc.metadata.get('title')} | page {doc.metadata.get('page')}")
    print(doc.page_content[:200], "\n---")