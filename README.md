# trustworthy-money-rag

A retrieval-augmented generation (RAG) service that answers questions about Islamic
finance, microfinance and charity transparency from a curated corpus of 16 academic
and institutional papers.

Built as the **system under test** for my AI-evaluation toolkit
([LLM_Eval](https://github.com/AwaisAhmadTools/LLM_Eval)): it exposes a stable HTTP
contract so the evaluation suite can score its retrieval and generation quality.

## Pipeline

    corpus/*.pdf ──loader──> 526 page-documents ──chunker──> 1,135 chunks
                 ──embed (text-embedding-3-small)──> Chroma (persistent, local)
                 ──retrieve top-3──> LLM (gpt-4o-mini, temp 0) ──> answer + evidence

## Layout

| Path | What it does |
|---|---|
| `rag_app/loader.py` | `load_corpus_documents()` — PyPDFLoader/TextLoader over `corpus/`, one Document per page |
| `rag_app/chunker.py` | `chunk_documents()` — token-based (tiktoken) splitting, 500 tokens / 75 overlap, sentence-aware separators |
| `rag_app/vectorstore.py` | `build_index()`, `load_index()`, `search()` — OpenAI embeddings + persistent Chroma collection |
| `rag_app/config.py` | Shared constants (`EXPECTED_CHUNKS`, `TEST_QUERY`) |
| `app.py` | FastAPI service exposing `POST /ask` |
| `smoke_test*.py` | Verification scripts (see Verification) |

## Setup

    python -m venv .venv
    .venv\Scripts\activate            # Windows
    pip install -r requirements.txt
    copy .env.example .env            # then add your OPENAI_API_KEY

Requires Python 3.13.

## The corpus

The PDFs are **not** in this repo — they are third-party papers kept locally and
excluded via `.gitignore`. `corpus_manifest.md` lists all 16 with source URLs.
Download them into `corpus/` before building the index.

## Build the index (once)

    python smoke_test_vectorstore.py

Creates `chroma_db/` if absent, embeds the chunks (roughly 1p of API usage) and
asserts the record count. Re-running does **not** rebuild unless `chroma_db/` is missing.

## Run the service

    uvicorn app:app --reload --port 8000

- Interactive API docs: http://localhost:8000/docs
- Health check: http://localhost:8000/

## API contract

    POST /ask
    Request:   {"question": "What is murabaha?", "chat_history": []}
    Response:  {"answer": "...",
                "retrieved_docs": [{"page_content": "...", "score": 0.64}, ...]}

Three retrieved documents per request (k=3). The retrieved evidence is returned
alongside the answer so retrieval quality and faithfulness are measurable.

## Verification

| Script | What it checks |
|---|---|
| `smoke_test.py` | Loader: document count, per-file char counts, pages with no extractable text |
| `smoke_test_vectorstore.py` | Builds or reuses the index, asserts 1,135 records, runs a query |
| `smoke_test_verify_persistence.py` | Loads the index from disk (no re-embedding) and asserts the same record count |

## Data quality notes

Notes on the source material, since they affect retrieval and how metrics should be read.

- **Filename-derived labels.** Chunks carry a `source_file` label taken from the filename.
  PDF embedded `title` metadata is unreliable (often blank, occasionally an export artifact
  such as an InDesign filename), so it isn't used for display.
- **Some pages yield no text.** 3 of 526 pages in the current corpus produce no extractable
  text (typically image-only pages). Those papers therefore have small gaps.
- **Figure and table text can extract as noise.** Diagram labels sometimes decode as garbled
  characters because of embedded font mappings — so a small number of chunks contain
  non-prose noise, which can affect retrieval and judge scoring.

## Known limitations

- **Chunking is an opening configuration, not a tuned one** — size/overlap/separators
  were chosen deliberately but will be tuned once retrieval metrics exist to measure them.
- **Page-level splitting** — chunks do not flow across PDF page breaks.
- **Index builds are not idempotent** — re-running a build without deleting `chroma_db/`
  would duplicate records (IDs are not yet stable). Guarded by an existence check plus a
  count assertion.
- **`chat_history` is accepted but unused** — contract compatibility only; single-turn today.
- **The answer model (gpt-4o-mini) is deliberately weaker than the judge model (gpt-4o)** used by the evaluation suite, to avoid self-preference bias in scoring.

