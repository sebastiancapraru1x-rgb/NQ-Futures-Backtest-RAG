"""
build_index.py — One-time setup: ingest CSV → embed → build FAISS index.
Run this ONCE before using app.py.
Downloads the sentence-transformers embedding model here (~80MB).
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from ingest_documents import ingest_documents, save_chunks
from embeddings import HybridEmbedder
from vector_store import VectorStore


def main():
    base = Path(__file__).parent
    csv_path = base / "data" / "sample_backtests.csv"
    index_path = base / "vector_store_index"
    chunks_path = base / "data" / "chunks.json"

    print("=" * 60)
    print("Building NQ Backtest RAG Index")
    print("=" * 60)

    # Step 1: Ingest CSV → chunks + metadata (no model download)
    print("\n[1/4] Ingesting backtest CSV...")
    chunks, metadatas = ingest_documents(str(csv_path))
    save_chunks(chunks, metadatas, str(chunks_path))

    # Step 2: Create embeddings (DOWNLOADS sentence-transformers model here)
    print("\n[2/4] Creating embeddings (downloading model if needed)...")
    embedder = HybridEmbedder()
    embedder.fit_sparse(chunks)                          # TF-IDF fit (no download)
    dense_embs = embedder.encode_dense(chunks)           # PyTorch model download happens here
    sparse_embs = embedder.encode_sparse(chunks)
    print(f"    Dense shape: {dense_embs.shape}")
    print(f"    Sparse shape: {sparse_embs.shape}")

    # Step 3: Build FAISS vector store
    print("\n[3/4] Building FAISS index...")
    store = VectorStore(embedding_dim=dense_embs.shape[1])
    store.add(dense_embs, chunks, metadatas)
    store.set_sparse_embeddings(sparse_embs)

    # Step 4: Save to disk
    print("\n[4/4] Saving index to disk...")
    store.save(str(index_path))

    print("\n" + "=" * 60)
    print("Index built successfully!")
    print(f"Location: {index_path}")
    print("Now run: python app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()