from rag_app.loader import load_corpus_documents
from rag_app.chunker import chunk_documents, token_length

docs = load_corpus_documents()
chunks = chunk_documents(docs)

# Test the document loader - currently split by page
print(f"{len(docs)} documents loaded")
for doc in docs [:5]:
    print(f" {doc.metadata['source']} -> {len(doc.page_content)} chars")

# test the chunker
chars = [len(c.page_content) for c in chunks]
toks  = [token_length(c.page_content) for c in chunks]
print(f"docs: {len(docs)}  →  chunks: {len(chunks)}")
print(f"tokens — min: {min(toks)}  avg: {sum(toks)//len(toks)}  max: {max(toks)}")
print(f"chars  — min: {min(chars)} avg: {sum(chars)//len(chars)} max: {max(chars)}")

mid_sentence = [c for c in chunks if c.page_content.rstrip()[-1:] not in ".!?\"'"]
print(f"chunks ending mid-sentence: {len(mid_sentence)} ({100*len(mid_sentence)//len(chunks)}%)")

print("\n--- sample chunk ---")
print(chunks[0].metadata)
print(chunks[0].page_content[:400])

empty_pages = [d for d in docs if not d.page_content.strip()]
print(f"pages with no extractable text: {len(empty_pages)}")