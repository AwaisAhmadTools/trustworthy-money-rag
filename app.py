"""FastAPI service exposing the RAG pipeline over HTTP.
    POST /ask   {question, chat_history}  ->  {answer, retrieved_docs: [{page_content, score}]}
"""

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI                          # web framework: routing, JSON, validation
from pydantic import BaseModel                     # data contracts (DTOs) with runtime validation
from langchain_openai import ChatOpenAI                  # the generator (chat model wrapper)
from langchain_core.messages import SystemMessage, HumanMessage

from rag_app.vectorstore import search

# Answer Model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# This prompt is for the faithfulness measure. A permissive prompt would
# make the metric meaningless — the model could answer from its own knowledge.
SYSTEM_PROMPT = (
    "You answer questions using ONLY the provided context. "
    "If the context does not contain the answer, reply exactly: "
    "'I don't know based on the provided context.' "
    "Do not use outside knowledge. Do not speculate."
)

app = FastAPI(title="trustworthy-money-rag")

class AskRequest(BaseModel):
    question: str
    chat_history: list = [] 

class RetrievedDoc(BaseModel):
    page_content: str
    score: float

class AskResponse(BaseModel):
    answer: str
    retrieved_docs: list[RetrievedDoc]

# --- Health check -----------------------------------------------------------
@app.get("/")
def health() -> dict:
    return {"status": "ok"}

# --- The endpoint -----------------------------------------------------------
@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    # 1. RETRIEVE — top-3 chunks from the Chroma index
    results = search(request.question, k=3)

    # 2. AUGMENT — stitch the chunks into one context block
    context = "\n\n---\n\n".join(doc.page_content for doc, _ in results)

    # 3. GENERATE — the LLM answers strictly from that context
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {request.question}"),
    ]
    answer = llm.invoke(messages).content

    return AskResponse(
        answer=answer,
        retrieved_docs=[
            RetrievedDoc(page_content=doc.page_content, score=float(score))
            for doc, score in results
        ],
    )