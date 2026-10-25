"""Knowledge Hub API: upload company documents and ask questions answered from them (RAG)."""
from __future__ import annotations

import re
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .ai import AIClient, AIError
from .config import Settings
from .seeds import SEED_DOCUMENTS
from .store import Store
from .text import SUPPORTED_EXTENSIONS, ParseError, chunk_text, extract_text, keyword_scores

SYSTEM_PROMPT = (
    "You are Knowledge Hub, an enterprise document assistant. Answer only from the retrieved company knowledge. "
    "If the sources do not contain enough evidence, say so clearly and ask the user to narrow the question. "
    "Do not invent policies, dates, names, or numbers. Keep the answer concise and practical. "
    "Use [S1], [S2], etc. inline when a claim is supported by a source."
)
SECRET_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (r"sk-[A-Za-z0-9_-]{20,}", r"password\s*[:=]", r"secret\s*[:=]", r"api[_ -]?key\s*[:=]", r"AKIA[0-9A-Z]{16}")
]
CLIENT_ID = re.compile(r"^[A-Za-z0-9-]{8,64}$")


class HistoryTurn(BaseModel):
    role: str = Field(default="user", max_length=20)
    content: str = Field(default="", max_length=4000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    history: list[HistoryTurn] = Field(default_factory=list, max_length=12)


class RateLimiter:
    """In-memory sliding window per client IP. Good enough for a single-instance demo."""

    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str, limit: int, window: float = 3600.0) -> None:
        if limit <= 0:
            return
        now = time.monotonic()
        with self._lock:
            hits = self._hits[key]
            while hits and now - hits[0] > window:
                hits.popleft()
            if len(hits) >= limit:
                wait = int(window - (now - hits[0])) // 60 + 1
                raise HTTPException(status_code=429, detail=f"This demo allows {limit} requests per hour. Try again in about {wait} minutes.")
            hits.append(now)


def create_app(settings: Settings | None = None, ai: AIClient | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    ai = ai or AIClient(settings)
    store = Store(settings.db_path)
    limiter = RateLimiter()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        store.init(SEED_DOCUMENTS, chunk_text)
        yield

    app = FastAPI(title="Knowledge Hub API", version="2.0.0", lifespan=lifespan)
    app.state.settings, app.state.store, app.state.ai = settings, store, ai
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type", "X-Client-Id"],
    )

    def client_ip(request: Request) -> str:
        if settings.trust_proxy:
            forwarded = request.headers.get("x-forwarded-for", "")
            if forwarded:
                return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def valid_client(client_id: str | None) -> str | None:
        return client_id if client_id and CLIENT_ID.match(client_id) else None

    # ---------- retrieval ----------
    def ensure_vectors(rows) -> None:
        missing = [row for row in rows if store.vector_for(row, settings.embedding_model) is None]
        if missing:
            vectors = ai.embed([row["content"] for row in missing])
            store.save_vectors([(row["id"], v) for row, v in zip(missing, vectors)], settings.embedding_model)

    def search(question: str, limit: int = 4) -> tuple[list[dict[str, Any]], str, str | None]:
        """Returns (sources, method, notice). Falls back to keyword search if embeddings fail."""
        rows = store.all_chunks()
        if not rows:
            return [], "none", None
        method, notice = "keyword", None
        scores: list[float] | None = None
        if ai.enabled:
            try:
                ensure_vectors(rows)
                query = ai.embed([question])[0]
                query_norm = sum(v * v for v in query) ** 0.5 or 1.0
                scores = []
                for row in rows:
                    vector, norm = store.vector_for(row, settings.embedding_model)  # type: ignore[misc]
                    dot = sum(a * b for a, b in zip(query, vector))
                    scores.append(dot / (query_norm * norm) if norm else 0.0)
                method = "semantic"
            except AIError as exc:
                notice = exc.message
        if scores is None:
            scores = keyword_scores(question, [row["content"] for row in rows])
        ranked = sorted(zip(scores, rows), key=lambda item: item[0], reverse=True)
        sources = [
            {
                "score": round(max(0.0, min(0.999, score)), 3),
                "document": row["document_name"],
                "filename": row["filename"],
                "chunk_index": row["chunk_index"],
                "content": row["content"],
            }
            for score, row in ranked[:limit]
            if score > 0
        ]
        return sources, method, notice

    def passages_answer(sources: list[dict[str, Any]], reason: str) -> str:
        lines = [reason, ""]
        for i, source in enumerate(sources[:3], start=1):
            excerpt = re.sub(r"\s+", " ", source["content"]).strip()
            if len(excerpt) > 360:
                excerpt = excerpt[:360].rsplit(" ", 1)[0] + "…"
            lines.append(f"**{source['document']}** [S{i}]\n{excerpt}\n")
        return "\n".join(lines).strip()

    # ---------- routes ----------
    @app.get("/", include_in_schema=False)
    def serve_app() -> FileResponse:
        return FileResponse(settings.frontend_file)

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        ready = ai.ready()
        return {
            "api_configured": ready,
            "mode": "ai" if ready else "retrieval_only",
            "provider": settings.provider,
            "provider_label": settings.provider_label,
            "provider_configured": ai.enabled,
            "setup_hint": None if ready else settings.setup_hint(),
            "chat_model": settings.chat_model or None,
            "embedding_model": settings.embedding_model or None,
            "indexed_chunks": len(store.all_chunks()),
            "demo_mode": settings.demo_mode,
            "max_upload_mb": settings.max_upload_mb,
        }

    @app.get("/api/documents")
    def list_documents() -> list[dict[str, Any]]:
        return store.list_documents()

    @app.post("/api/documents")
    async def upload_document(request: Request, file: UploadFile = File(...)) -> dict[str, Any]:
        limiter.check(f"upload:{client_ip(request)}", settings.uploads_per_hour)
        filename = Path(file.filename or "untitled").name
        if Path(filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise HTTPException(status_code=400, detail="Supported formats are PDF, DOCX, Markdown and plain text.")
        limit = settings.max_upload_mb * 1024 * 1024
        raw = await file.read(limit + 1)
        if len(raw) > limit:
            raise HTTPException(status_code=413, detail=f"File size must be {settings.max_upload_mb} MB or less.")
        try:
            chunks = chunk_text(extract_text(filename, raw))
        except ParseError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if not chunks:
            raise HTTPException(status_code=422, detail="No readable text was found in this file. Scanned PDFs need OCR first.")

        vectors, warning = None, None
        if ai.enabled:
            try:
                vectors = ai.embed(chunks)
            except AIError as exc:
                warning = f"Saved, but semantic search is unavailable for now: {exc.message}"
        name = re.sub(r"[_-]+", " ", Path(filename).stem).strip().title() or "Untitled"
        document = store.add_document(
            name=name,
            filename=filename,
            category="Uncategorized",
            size_bytes=len(raw),
            chunks=chunks,
            vectors=vectors,
            model=settings.embedding_model if vectors else None,
        )
        store.prune_uploads(settings.max_documents)
        if warning:
            document["warning"] = warning
        return document

    @app.delete("/api/documents/{document_id}")
    def delete_document(document_id: str) -> dict[str, Any]:
        row = store.get_document(document_id)
        if row is None:
            raise HTTPException(status_code=404, detail="That document no longer exists.")
        if settings.demo_mode and row["is_seed"]:
            raise HTTPException(status_code=403, detail="Sample documents can't be deleted in the demo.")
        store.delete_document(document_id)
        return {"deleted": document_id}

    @app.post("/api/chat")
    def chat(body: ChatRequest, request: Request, x_client_id: str | None = Header(default=None)) -> dict[str, Any]:
        limiter.check(f"chat:{client_ip(request)}", settings.chat_per_hour)
        started = time.perf_counter()
        client_id = valid_client(x_client_id)
        question = body.question.strip()
        if any(p.search(question) for p in SECRET_PATTERNS):
            return {
                "answer": "I can't process secrets, passwords, or API keys. Please remove sensitive values and ask about the business policy or workflow instead.",
                "sources": [],
                "confidence": 0,
                "filtered": True,
                "mode": "filtered",
            }
        sources, method, notice = search(question)
        if not store.list_documents():
            raise HTTPException(status_code=404, detail="No knowledge has been indexed yet. Upload a document first.")
        if not sources:
            answer = "I couldn't find anything about that in the indexed documents. Try different keywords or upload a document that covers it."
            store.add_question(question, answer, 0.0, "no_match", client_id, int((time.perf_counter() - started) * 1000), [])
            return {"answer": answer, "sources": [], "confidence": 0, "filtered": False, "mode": "no_match"}

        confidence = round(sum(s["score"] for s in sources[:3]) / min(3, len(sources)), 3)
        mode = "ai"
        if ai.enabled and notice is None:
            context = "\n\n".join(
                f"[S{i}] {s['document']} (chunk {s['chunk_index'] + 1})\n{s['content']}" for i, s in enumerate(sources, start=1)
            )
            previous = "\n".join(f"{t.role}: {t.content[:600]}" for t in body.history[-4:])
            prompt = f"Question:\n{question}\n\nRetrieved company knowledge:\n{context}\n\nRecent conversation (optional):\n{previous or 'None'}"
            try:
                answer = ai.chat(SYSTEM_PROMPT, prompt)
            except AIError as exc:
                notice = exc.message
        if not ai.enabled:
            mode = "retrieval_only"
            answer = passages_answer(sources, "AI answers are turned off on this server, so here are the passages that best match your question:")
        elif notice is not None:
            mode = "retrieval_only"
            answer = passages_answer(sources, f"The AI model is unavailable right now ({notice}) Here are the most relevant passages instead:")

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        store.add_question(question, answer, confidence, mode, client_id, elapsed_ms, [s["document"] for s in sources])
        return {
            "answer": answer,
            "sources": [
                {
                    "document": s["document"],
                    "filename": s["filename"],
                    "chunk_index": s["chunk_index"],
                    "score": s["score"],
                    "preview": s["content"][:220],
                }
                for s in sources
            ],
            "confidence": confidence,
            "filtered": False,
            "mode": mode,
            "retrieval": method,
            "response_ms": elapsed_ms,
        }

    @app.get("/api/history")
    def history(x_client_id: str | None = Header(default=None)) -> list[dict[str, Any]]:
        # Locally you see every question; in the public demo each visitor only sees their own.
        return store.history(valid_client(x_client_id), include_unowned=not settings.demo_mode)

    @app.get("/api/stats")
    def stats() -> dict[str, Any]:
        return store.stats()

    return app


app = create_app()
