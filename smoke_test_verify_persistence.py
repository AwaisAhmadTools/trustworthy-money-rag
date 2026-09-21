from dotenv import load_dotenv
load_dotenv()

from rag_app.vectorstore import load_index, search
from rag_app.config import EXPECTED_CHUNKS, TEST_QUERY


store = load_index()                      # no build, no document embedding
count = store._collection.count()
assert count == EXPECTED_CHUNKS, (
    f"index empty or wrong: {count} records (expected {EXPECTED_CHUNKS}) — rebuild via smoke_test_vectorstore.py"
)
print(f"records: {count} ✓ (loaded from disk, no re-embedding)")

for doc, score in search(TEST_QUERY, k=3):
    print(f"{score:.3f} | {doc.metadata.get('title')} | page {doc.metadata.get('page')}")