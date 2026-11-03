"""FastAPI interface for document ingestion and question answering."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from .pipeline import RAGPipeline

app = FastAPI(title="RAG Engine", version="0.1.0")
pipeline = RAGPipeline()


class IngestRequest(BaseModel):
    directory: str = "data"


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=20)


@app.get("/health")
def health():
    return {"status": "ok", "service": "rag-engine"}


@app.post("/ingest")
def ingest(request: IngestRequest):
    try:
        return pipeline.ingest(request.directory)
    except (ValueError, NotADirectoryError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/ask")
def ask(request: AskRequest):
    try:
        return pipeline.ask(request.question, request.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
