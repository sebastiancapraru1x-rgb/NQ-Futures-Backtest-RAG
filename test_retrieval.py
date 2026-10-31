from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from ingest_documents import ingest_documents
from embeddings import HybridEmbedder
from vector_store import VectorStore

# Load existing index
store = VectorStore.load("vector_store_index")

# Initialize embedder (loads cached model)
embedder = HybridEmbedder()

queries = [
    "What was the win rate for IFVG_CISD_EMA in 2025-2026?",
    "Which months were best and worst?",
    "How many trades were taken in 2023-2024?",
]

for q in queries:
    print(f"\n{'='*60}")
    print(f"QUERY: {q}")
    print('='*60)
    
    q_vec = embedder.dense.encode_single(q)
    results = store.search(q_vec, top_k=3)
    
    for i, r in enumerate(results, 1):
        meta = r['metadata']
        print(f"\n[{i}] Score: {r['score']:.3f}")
        print(f"    Strategy: {meta.get('strategy', 'N/A')}")
        print(f"    Years: {meta.get('years', 'N/A')}")
        print(f"    Type: {meta.get('chunk_type', 'N/A')}")
        print(f"    Text: {r['document'][:200]}...")