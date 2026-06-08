"""
embedder.py
-----------
Takes crawled pages, splits them into overlapping chunks,
embeds them with sentence-transformers, and stores them in ChromaDB.
"""

import hashlib
import logging
from typing import List, Dict

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

# Local embedding model — no API key needed, runs on CPU
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DB_PATH = "./data/chroma_db"
COLLECTION_NAME = "website_chunks"


def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """
    Split text into overlapping chunks of approximately `chunk_size` characters.
    Splits on sentence boundaries when possible.
    """
    # Split into sentences (simple heuristic)
    sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]

    chunks: List[str] = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) + 2 <= chunk_size:
            current += sentence + ". "
        else:
            if current:
                chunks.append(current.strip())
            # Start new chunk with overlap from previous
            words = current.split()
            overlap_text = " ".join(words[-overlap // 5 :]) if words else ""
            current = overlap_text + " " + sentence + ". "

    if current.strip():
        chunks.append(current.strip())

    return [c for c in chunks if len(c) > 50]  # Filter very short chunks


def _make_doc_id(url: str, chunk_index: int) -> str:
    """Create a stable unique ID for a chunk."""
    raw = f"{url}__chunk__{chunk_index}"
    return hashlib.md5(raw.encode()).hexdigest()


class Embedder:
    """Handles chunking, embedding, and storing documents in ChromaDB."""

    def __init__(self, db_path: str = CHROMA_DB_PATH):
        logger.info("Loading embedding model: %s", EMBED_MODEL_NAME)
        self.model = SentenceTransformer(EMBED_MODEL_NAME)

        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("ChromaDB collection ready. Items: %d", self.collection.count())

    def index_pages(self, pages: List[Dict[str, str]], progress_callback=None) -> int:
        """
        Chunk, embed, and store a list of crawled pages.

        Parameters
        ----------
        pages : list of dicts with keys 'url', 'title', 'text'
        progress_callback : callable(current, total) — optional UI progress

        Returns
        -------
        int : total number of chunks indexed
        """
        all_ids: List[str] = []
        all_embeddings: List[List[float]] = []
        all_documents: List[str] = []
        all_metadatas: List[Dict] = []

        total_pages = len(pages)
        for page_idx, page in enumerate(pages):
            chunks = _chunk_text(page["text"])
            for chunk_idx, chunk in enumerate(chunks):
                doc_id = _make_doc_id(page["url"], chunk_idx)
                all_ids.append(doc_id)
                all_documents.append(chunk)
                all_metadatas.append(
                    {"url": page["url"], "title": page["title"], "chunk_index": chunk_idx}
                )

            if progress_callback:
                progress_callback(page_idx + 1, total_pages)

        if not all_ids:
            logger.warning("No chunks to index.")
            return 0

        logger.info("Embedding %d chunks...", len(all_ids))
        embeddings = self.model.encode(
            all_documents, show_progress_bar=True, batch_size=32
        ).tolist()

        # Upsert in batches of 500 to avoid memory issues
        batch_size = 500
        for i in range(0, len(all_ids), batch_size):
            self.collection.upsert(
                ids=all_ids[i : i + batch_size],
                embeddings=embeddings[i : i + batch_size],
                documents=all_documents[i : i + batch_size],
                metadatas=all_metadatas[i : i + batch_size],
            )

        logger.info("Indexed %d chunks successfully.", len(all_ids))
        return len(all_ids)

    def reset(self):
        """Delete all indexed documents."""
        self.client.delete_collection(COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Collection reset.")

    @property
    def count(self) -> int:
        return self.collection.count()
