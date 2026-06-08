"""
retriever.py
------------
Queries ChromaDB to find the most relevant chunks for a given question.
"""

import logging
from typing import List, Dict

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

from .embedder import EMBED_MODEL_NAME, CHROMA_DB_PATH, COLLECTION_NAME

logger = logging.getLogger(__name__)


class Retriever:
    """Retrieves relevant chunks from ChromaDB for a query."""

    def __init__(self, db_path: str = CHROMA_DB_PATH):
        self.model = SentenceTransformer(EMBED_MODEL_NAME)
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Find the top-k most relevant chunks for the given query.

        Returns
        -------
        list of dicts: {text, url, title, score}
        """
        if self.collection.count() == 0:
            return []

        query_embedding = self.model.encode([query]).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunks.append(
                {
                    "text": doc,
                    "url": meta.get("url", ""),
                    "title": meta.get("title", ""),
                    "score": round(1 - dist, 4),  # cosine similarity
                }
            )

        return chunks

    def build_context(self, chunks: List[Dict]) -> str:
        """Format retrieved chunks into a single context string for the LLM."""
        parts = []
        for i, chunk in enumerate(chunks, 1):
            parts.append(f"[Source {i}: {chunk['title']} — {chunk['url']}]\n{chunk['text']}")
        return "\n\n---\n\n".join(parts)

    @property
    def count(self) -> int:
        return self.collection.count()
