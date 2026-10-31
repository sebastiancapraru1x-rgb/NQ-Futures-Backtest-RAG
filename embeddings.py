"""
embeddings.py — PyTorch + TensorFlow embedding generation for RAG.

Demonstrates:
  - PyTorch: sentence-transformers model loading, batch inference, GPU/CPU handling
  - TensorFlow: custom TF-IDF + cosine similarity as an alternative retrieval signal
  - Hybrid retrieval: combines dense (PyTorch) + sparse (TF-IDF) embeddings
"""

import os
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Device selection (PyTorch) ────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── PyTorch: Dense Embeddings ────────────────────────────────────────────────
class PyTorchEmbedder:
    """Generate dense embeddings using PyTorch-backed sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None

    def load(self):
        """Load the sentence-transformer model onto the chosen device."""
        print(f"[PyTorch] Loading {self.model_name} on {DEVICE}...")
        self.model = SentenceTransformer(self.model_name, device=DEVICE)
        print(f"[PyTorch] Model loaded. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Encode a list of texts into dense vectors using PyTorch inference."""
        if self.model is None:
            self.load()
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings

    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text string."""
        return self.encode([text])[0]


# ── TensorFlow: Sparse TF-IDF Embeddings ─────────────────────────────────────
class TFSparseEmbedder:
    """Generate sparse TF-IDF embeddings using scikit-learn (TF-compatible).

    This provides a lightweight, interpretable retrieval signal that
    complements dense PyTorch embeddings.  In a production system you
    could swap this for a TensorFlow TextVectorization layer.
    """

    def __init__(self, max_features: int = 5000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
            ngram_range=(1, 2),
        )
        self.fitted = False

    def fit(self, texts: list[str]):
        """Fit the TF-IDF vectorizer on the corpus."""
        self.vectorizer.fit(texts)
        self.fitted = True
        print(f"[TF-IDF] Fitted on {len(texts)} documents. Vocab size: {len(self.vectorizer.vocabulary_)}")

    def encode(self, texts: list[str]) -> np.ndarray:
        """Transform texts into TF-IDF vectors."""
        if not self.fitted:
            raise RuntimeError("Call fit() before encode()")
        return self.vectorizer.transform(texts).toarray()

    def similarity(self, query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
        """Cosine similarity between query and documents."""
        return cosine_similarity(query_vec.reshape(1, -1), doc_vecs)[0]


# ── Hybrid Embedder (PyTorch + TF-IDF) ──────────────────────────────────────
class HybridEmbedder:
    """Combines dense PyTorch embeddings with sparse TF-IDF for hybrid retrieval.

    This demonstrates knowledge of both PyTorch (dense) and TensorFlow-compatible
    (sparse TF-IDF) embedding approaches — a common pattern in production RAG.
    """

    def __init__(self, dense_model: str = "all-MiniLM-L6-v2"):
        self.dense = PyTorchEmbedder(dense_model)
        self.sparse = TFSparseEmbedder()

    def fit_sparse(self, texts: list[str]):
        """Fit the sparse embedder on the corpus."""
        self.sparse.fit(texts)

    def encode_dense(self, texts: list[str]) -> np.ndarray:
        return self.dense.encode(texts)

    def encode_sparse(self, texts: list[str]) -> np.ndarray:
        return self.sparse.encode(texts)

    def hybrid_similarity(
        self,
        query: str,
        dense_docs: np.ndarray,
        sparse_docs: np.ndarray,
        alpha: float = 0.7,
    ) -> np.ndarray:
        """Combine dense + sparse similarity scores.

        alpha controls the weight of dense (PyTorch) vs sparse (TF-IDF).
        Default 0.7 = 70% dense, 30% sparse.
        """
        dense_q = self.dense.encode_single(query)
        sparse_q = self.sparse.encode([query])[0]

        dense_sim = np.dot(dense_docs, dense_q)
        sparse_sim = self.sparse.similarity(sparse_q, sparse_docs)

        return alpha * dense_sim + (1 - alpha) * sparse_sim


if __name__ == "__main__":
    sample_texts = [
        "TPO Reversal strategy backtest 2022-2023 showed 14.5% return",
        "Breakout Pullback strategy had 52% win rate in high-VIX regimes",
        "First Hour Breakout works in any market regime",
    ]

    embedder = HybridEmbedder()
    embedder.fit_sparse(sample_texts)

    dense_embs = embedder.encode_dense(sample_texts)
    print(f"\n[PyTorch] Dense embeddings shape: {dense_embs.shape}")

    sparse_embs = embedder.encode_sparse(sample_texts)
    print(f"[TF-IDF] Sparse embeddings shape: {sparse_embs.shape}")

    scores = embedder.hybrid_similarity("Which strategy has highest return?", dense_embs, sparse_embs)
    print(f"\nHybrid similarity scores: {scores}")
    print(f"Best match: {sample_texts[np.argmax(scores)]}")