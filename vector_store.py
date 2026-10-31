"""
vector_store.py — FAISS-based vector store with metadata filtering.

Demonstrates:
  - FAISS index creation (FlatIP for cosine similarity)
  - Metadata filtering by strategy name, years, metrics
  - Save/load index to disk
  - Hybrid retrieval (dense + sparse score fusion)
"""

import os
import pickle
import numpy as np
import faiss
from typing import Optional


class VectorStore:
    """FAISS vector store with metadata filtering and hybrid retrieval."""

    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.documents: list[str] = []
        self.metadatas: list[dict] = []
        self.sparse_embeddings: Optional[np.ndarray] = None

    def add(self, embeddings: np.ndarray, documents: list[str], metadatas: list[dict]):
        """Add embeddings + documents + metadata to the store."""
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(f"Expected dim {self.embedding_dim}, got {embeddings.shape[1]}")
        self.index.add(embeddings)
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)
        print(f"[VectorStore] Added {len(documents)} documents. Total: {self.index.ntotal}")

    def set_sparse_embeddings(self, sparse_embs: np.ndarray):
        """Store sparse (TF-IDF) embeddings for hybrid retrieval."""
        self.sparse_embeddings = sparse_embs

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filter_strategy: Optional[str] = None,
        filter_years: Optional[str] = None,
    ) -> list[dict]:
        """Search the vector store with optional metadata filtering."""
        if self.index.ntotal == 0:
            return []

        D, I = self.index.search(query_embedding.reshape(1, -1), top_k * 3)

        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self.documents):
                continue

            meta = self.metadatas[idx]

            if filter_strategy and meta.get("strategy") != filter_strategy:
                continue
            if filter_years and meta.get("years") != filter_years:
                continue

            results.append({
                "document": self.documents[idx],
                "metadata": meta,
                "score": float(score),
                "index": int(idx),
            })

            if len(results) >= top_k:
                break

        return results

    def hybrid_search(
        self,
        query: str,
        hybrid_embedder,
        top_k: int = 5,
        alpha: float = 0.7,
        filter_strategy: Optional[str] = None,
        filter_years: Optional[str] = None,
    ) -> list[dict]:
        """Hybrid search combining dense (PyTorch) + sparse (TF-IDF) scores."""
        if self.sparse_embeddings is None:
            raise RuntimeError("Sparse embeddings not set. Call set_sparse_embeddings() first.")

        dense_docs = np.array([self.index.reconstruct(i) for i in range(self.index.ntotal)])
        combined_scores = hybrid_embedder.hybrid_similarity(
            query, dense_docs, self.sparse_embeddings, alpha=alpha
        )

        scored_results = []
        for idx, score in enumerate(combined_scores):
            meta = self.metadatas[idx]
            if filter_strategy and meta.get("strategy") != filter_strategy:
                continue
            if filter_years and meta.get("years") != filter_years:
                continue
            scored_results.append({
                "document": self.documents[idx],
                "metadata": meta,
                "score": float(score),
                "index": idx,
            })

        scored_results.sort(key=lambda x: x["score"], reverse=True)
        return scored_results[:top_k]

    def save(self, path: str):
        """Save index + documents + metadata to disk."""
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "documents.pkl"), "wb") as f:
            pickle.dump(self.documents, f)
        with open(os.path.join(path, "metadatas.pkl"), "wb") as f:
            pickle.dump(self.metadatas, f)
        if self.sparse_embeddings is not None:
            np.save(os.path.join(path, "sparse_embeddings.npy"), self.sparse_embeddings)
        print(f"[VectorStore] Saved to {path}")

    @classmethod
    def load(cls, path: str) -> "VectorStore":
        """Load index + documents + metadata from disk."""
        index = faiss.read_index(os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "documents.pkl"), "rb") as f:
            documents = pickle.load(f)
        with open(os.path.join(path, "metadatas.pkl"), "rb") as f:
            metadatas = pickle.load(f)

        store = cls(embedding_dim=index.d)
        store.index = index
        store.documents = documents
        store.metadatas = metadatas

        sparse_path = os.path.join(path, "sparse_embeddings.npy")
        if os.path.exists(sparse_path):
            store.sparse_embeddings = np.load(sparse_path)

        print(f"[VectorStore] Loaded {store.index.ntotal} documents from {path}")
        return store

    def __len__(self):
        return self.index.ntotal


if __name__ == "__main__":
    store = VectorStore(embedding_dim=4)
    embs = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]], dtype=np.float32)
    docs = ["Strategy A backtest", "Strategy B backtest"]
    metas = [{"strategy": "A", "years": "2022-2023"}, {"strategy": "B", "years": "2022-2023"}]
    store.add(embs, docs, metas)
    print(f"Store size: {len(store)}")
    store.save("/tmp/test_vs")
    loaded = VectorStore.load("/tmp/test_vs")
    print(f"Loaded size: {len(loaded)}")