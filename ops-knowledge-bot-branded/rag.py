from __future__ import annotations
import os
import re
from dataclasses import dataclass
from typing import List, Dict, Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langdetect import detect
from dotenv import load_dotenv

# Load environment variables (if present)
load_dotenv()

# Configurable names for storage directory and collection
CHROMA_DIR = os.getenv("CHROMA_DIR", "./storage")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "ops_knowledge")

# Singleton holders for expensive resources
_model = None  # type: SentenceTransformer | None
_client = None  # type: chromadb.PersistentClient | None
_collection = None  # type: chromadb.api.models.Collection | None


def get_embedder() -> SentenceTransformer:
    """Lazily load the embedding model.  The model is cached for reuse."""
    global _model
    if _model is None:
        # This downloads the model the first time and caches it.
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _model


def get_chroma() -> tuple[chromadb.PersistentClient, chromadb.api.models.Collection]:
    """Return a persistent Chroma client and collection, creating them if needed."""
    global _client, _collection
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(allow_reset=False))
    if _collection is None:
        _collection = _client.get_or_create_collection(CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"})
    return _client, _collection


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a list of strings into vector embeddings using the shared model."""
    model = get_embedder()
    # Normalize embeddings so distance is cosine similarity distance
    return model.encode(texts, normalize_embeddings=True).tolist()


@dataclass
class RetrievedChunk:
    """Simple container for a retrieved document chunk."""
    text: str
    score: float
    metadata: Dict[str, Any]


def search(query: str, k: int = 6) -> List[RetrievedChunk]:
    """Perform a semantic search on the embedded documents and return the top k chunks."""
    _, col = get_chroma()
    qemb = embed_texts([query])[0]
    res = col.query(
        query_embeddings=[qemb],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    chunks: List[RetrievedChunk] = []
    docs = res.get("documents", [[]])[0]
    mds = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    for d, md, dist in zip(docs, mds, dists):
        # Convert distance to a similarity-like score (1.0 = perfect match)
        score = 1 - dist if dist is not None else 0.0
        chunks.append(RetrievedChunk(text=d, score=score, metadata=md))
    return chunks


def detect_lang(text: str) -> str:
    """Detect the language of a given text.  Defaults to English on error."""
    try:
        return detect(text)
    except Exception:
        return "en"


def format_citation(md: Dict[str, Any]) -> str:
    """Format a citation from metadata as `filename (p.X)`."""
    src = md.get("source", "unknown")
    page = md.get("page")
    if page:
        return f"{src} (p.{page})"
    return src


def build_extractive_answer(query: str, hits: List[RetrievedChunk], max_chars: int = 1200) -> Dict[str, Any]:
    """
    Build a bullet‑point extractive answer from retrieved chunks.  Returns a dict with
    the answer text, citations and the number of chunks used.

    This is used when no OpenAI API key is provided; it simply collects the best
    matching snippets and formats them with citations.
    """
    answer_lines: List[str] = []
    citations: List[str] = []
    current_len = 0
    for chunk in hits:
        snippet = chunk.text.strip()
        cite = format_citation(chunk.metadata)
        line = f"- {snippet}\n  — {cite}"
        if current_len + len(line) > max_chars:
            break
        answer_lines.append(line)
        citations.append(cite)
        current_len += len(line)

    # Remove duplicates but preserve order
    citations = list(dict.fromkeys(citations))

    lang = detect_lang(query)
    header = "Answer (extractive):" if lang != "ar" else "الإجابة (مقتبسة):"
    body = "\n\n".join(answer_lines) if answer_lines else ("No results found." if lang != "ar" else "لا توجد نتائج.")

    return {
        "mode": "extractive",
        "text": header + "\n\n" + body,
        "citations": citations,
        "k": len(hits),
    }