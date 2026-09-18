from rag_app.loader import load_corpus_documents

docs = load_corpus_documents()
print(f"{len(docs)} documents loaded")
for doc in docs [:5]:
    print(f" {doc.metadata['source']} -> {len(doc.page_content)} chars")