import os
import io
import logging
from typing import Optional

import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
import anthropic

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="WiredSocially RAG Voice Assistant", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

doc_chunks: list[str] = []
doc_embeddings: Optional[np.ndarray] = None
embed_model = None
doc_name: str = ""

client = anthropic.Anthropic()


def get_embed_model():
    global embed_model
    if embed_model is None:
        logger.info("Loading embedding model (first-time download may take a moment)...")
        from sentence_transformers import SentenceTransformer
        embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Embedding model loaded.")
    return embed_model


def extract_text(content: bytes, filename: str) -> str:
    name = filename.lower()
    if name.endswith(".txt") or name.endswith(".md"):
        return content.decode("utf-8", errors="replace")
    elif name.endswith(".pdf"):
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(content))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    elif name.endswith(".docx"):
        import docx
        doc = docx.Document(io.BytesIO(content))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    else:
        raise ValueError(f"Unsupported file type. Please upload PDF, TXT, MD, or DOCX.")


def chunk_text(text: str, chunk_size: int = 350, overlap: int = 60) -> list[str]:
    text = " ".join(text.split())
    words = text.split()

    if not words:
        return []

    if len(words) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
        if end == len(words):
            break

    return chunks


def cosine_similarity(query_emb: np.ndarray, doc_embs: np.ndarray) -> np.ndarray:
    q_norm = query_emb / (np.linalg.norm(query_emb) + 1e-10)
    d_norms = doc_embs / (np.linalg.norm(doc_embs, axis=1, keepdims=True) + 1e-10)
    return np.dot(d_norms, q_norm)


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    global doc_chunks, doc_embeddings, doc_name

    content = await file.read()

    try:
        text = extract_text(content, file.filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not text.strip():
        raise HTTPException(status_code=400, detail="Document appears to be empty or unreadable.")

    doc_name = file.filename
    doc_chunks = chunk_text(text)

    if not doc_chunks:
        raise HTTPException(status_code=400, detail="Could not extract text from document.")

    logger.info(f"Processing {len(doc_chunks)} chunks from '{doc_name}'...")
    model = get_embed_model()
    embeddings = model.encode(doc_chunks, show_progress_bar=False, batch_size=32)
    doc_embeddings = np.array(embeddings)
    logger.info("Document indexed successfully.")

    return {
        "status": "success",
        "filename": doc_name,
        "chunks": len(doc_chunks),
        "text_length": len(text),
    }


class QueryRequest(BaseModel):
    question: str


@app.post("/api/query")
async def query_document(request: QueryRequest):
    if not doc_chunks or doc_embeddings is None:
        raise HTTPException(status_code=400, detail="No document loaded. Please upload a document first.")

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    model = get_embed_model()
    q_emb = model.encode([request.question])[0]

    sims = cosine_similarity(q_emb, doc_embeddings)
    top_k = min(5, len(doc_chunks))
    top_indices = np.argsort(sims)[-top_k:][::-1]

    context_parts = []
    for i, idx in enumerate(top_indices):
        context_parts.append(f"[Excerpt {i + 1}]\n{doc_chunks[idx]}")
    context = "\n\n".join(context_parts)

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        system=(
            "You are a helpful, friendly voice assistant answering questions about an uploaded document. "
            "Keep answers concise (2-4 sentences), conversational, and clear — your response will be spoken aloud. "
            "If the answer isn't in the document, say so honestly."
        ),
        messages=[
            {
                "role": "user",
                "content": (
                    f"Using the following document excerpts, answer the question.\n\n"
                    f"Document excerpts:\n{context}\n\n"
                    f"Question: {request.question}"
                ),
            }
        ],
    )

    answer = response.content[0].text
    logger.info(f"Q: {request.question[:60]} | A: {answer[:60]}...")

    return {"answer": answer, "doc_name": doc_name}


@app.get("/api/status")
async def status():
    return {
        "ready": len(doc_chunks) > 0,
        "doc_name": doc_name,
        "chunks": len(doc_chunks),
        "tts": bool(os.getenv("OPENAI_API_KEY")),
    }


class TTSRequest(BaseModel):
    text: str
    voice: str = "nova"


@app.post("/api/tts")
async def text_to_speech(request: TTSRequest):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=400, detail="OPENAI_API_KEY not set.")

    from openai import OpenAI
    openai_client = OpenAI(api_key=api_key)

    audio = openai_client.audio.speech.create(
        model="tts-1",
        voice=request.voice,
        input=request.text,
    )
    return Response(content=audio.content, media_type="audio/mpeg")


# Serve frontend — mounted last so API routes always take priority
_frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(_frontend_dir):
    app.mount("/", StaticFiles(directory=_frontend_dir, html=True), name="static")
