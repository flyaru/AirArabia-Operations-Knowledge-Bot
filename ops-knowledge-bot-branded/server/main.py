from __future__ import annotations
import os
from typing import List, Dict, Any

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rag import search, build_extractive_answer, detect_lang

load_dotenv()

# Optional OpenAI integration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

if OPENAI_API_KEY:
    from openai import OpenAI

# Instantiate FastAPI application
app = FastAPI(title="Operations Knowledge Bot (Arwa & Air Arabia Edition)")


class AskRequest(BaseModel):
    question: str
    k: int = 6


class AskResponse(BaseModel):
    mode: str
    answer: str
    citations: List[str]
    used_k: int


@app.get("/health")
def health() -> Dict[str, bool]:
    """Health check endpoint."""
    return {"ok": True}


def synthesize_with_openai(question: str, contexts: List[Dict[str, Any]]) -> str:
    """Use OpenAI to synthesize a final answer from retrieved contexts."""
    client = OpenAI(api_key=OPENAI_API_KEY)
    system = (
        "You are an aviation operations assistant for ground operations staff. "
        "Answer concisely and politely, citing sources inline in square brackets with the format: [filename (p.X)]. "
        "If unsure, say so and suggest checking the relevant manual. "
        "Always answer in the same language as the question."
    )
    lang = detect_lang(question)
    # Build a context block for the LLM to reference
    context_block = "\n\n".join([
        f"Snippet {i+1} ({c['cite']}):\n{c['text']}" for i, c in enumerate(contexts)
    ])
    user = f"""Question ({lang}): {question}

Relevant snippets:
{context_block}

Write a concise answer (use bullets for procedures) and end with a short 'Key refs:' section listing the citations."
    # Call the model
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    """Handle a question by performing retrieval and optionally LLM synthesis."""
    hits = search(req.question, k=req.k)
    # Prepare context for synthesis
    contexts = [
        {
            "text": h.text.strip(),
            "cite": f"{h.metadata.get('source', '?')} (p.{h.metadata.get('page', '?')})",
        }
        for h in hits
    ]
    if OPENAI_API_KEY:
        answer = synthesize_with_openai(req.question, contexts)
        cits = sorted({c["cite"] for c in contexts})
        return AskResponse(mode="synthesized", answer=answer, citations=cits, used_k=len(hits))
    else:
        extractive = build_extractive_answer(req.question, hits)
        return AskResponse(
            mode=extractive["mode"],
            answer=extractive["text"],
            citations=extractive["citations"],
            used_k=extractive["k"],
        )


# --- Static Web UI ---

# Serve everything under /web as static files (CSS, images, etc.)
app.mount("/web", StaticFiles(directory="web"), name="web")


@app.get("/", response_class=HTMLResponse)
def root_index() -> str:
    """Serve the branded index.html as the root page."""
    index_path = os.path.join(os.path.dirname(__file__), "..", "web", "index.html")
    with open(index_path, "r", encoding="utf-8") as fh:
        return fh.read()