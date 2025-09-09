"""
Ingestion script for the operations knowledge bot.

This script extracts text from PDF files in the `data/` directory, splits the
text into overlapping chunks, embeds each chunk with sentence-transformers and
writes the embeddings to a persistent Chroma database.  Run this script
whenever you add or update manuals in the `data/` folder.
"""

import os
import re
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv
from tqdm import tqdm
from pypdf import PdfReader

from rag import get_chroma, embed_texts

load_dotenv()

# Paths
DATA_DIR = Path("data")
STORAGE_DIR = Path(os.getenv("CHROMA_DIR", "./storage"))
COLL_NAME = os.getenv("CHROMA_COLLECTION", "ops_knowledge")


def split_text(text: str, chunk_size: int = 1200, overlap: int = 150) -> List[str]:
    """Split a long string into overlapping chunks, attempting to break on sentence boundaries."""
    # Collapse whitespace
    clean = re.sub(r"\s+", " ", text).strip()
    chunks = []
    start = 0
    while start < len(clean):
        end = min(start + chunk_size, len(clean))
        # Try to end on a period after at least 300 characters
        dot = clean.rfind(".", start, end)
        if dot != -1 and dot > start + 300:
            end = dot + 1
        chunk = clean[start:end].strip()
        if chunk:
            chunks.append(chunk)
        # Overlap slightly to help context continuity
        start = max(0, end - overlap)
    return chunks


def extract_pdf(path: Path) -> List[Tuple[str, int]]:
    """Return a list of (text, page_number) tuples from the given PDF."""
    reader = PdfReader(str(path))
    pages: List[Tuple[str, int]] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        pages.append((text, i))
    return pages


def main() -> None:
    """Main entry point.  Ingest all PDFs in the data directory into Chroma."""
    _, col = get_chroma()

    pdfs = sorted(DATA_DIR.glob("*.pdf"))
    if not pdfs:
        print("No PDFs found in ./data. Add your manuals first.")
        return

    for pdf in pdfs:
        print(f"Ingesting: {pdf.name}")
        pages = extract_pdf(pdf)
        all_chunks: List[str] = []
        metadatas: List[dict] = []
        ids: List[str] = []

        for text, page_num in tqdm(pages, desc="Extract pages"):
            if not text.strip():
                continue
            chunks = split_text(text)
            for idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                metadatas.append({
                    "source": pdf.name,
                    "page": page_num,
                })
                ids.append(f"{pdf.name}-{page_num}-{idx}")

        # Insert in batches to avoid memory spikes
        BATCH = 128
        for i in range(0, len(all_chunks), BATCH):
            batch_docs = all_chunks[i:i + BATCH]
            batch_ids = ids[i:i + BATCH]
            batch_md = metadatas[i:i + BATCH]
            embs = embed_texts(batch_docs)
            col.upsert(ids=batch_ids, documents=batch_docs, metadatas=batch_md, embeddings=embs)

    print("Done. Collection is ready in ./storage.")


if __name__ == "__main__":
    main()