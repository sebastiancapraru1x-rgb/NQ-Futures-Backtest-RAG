# NQ Futures Backtest RAG

A Retrieval-Augmented Generation (RAG) system for NQ (Nasdaq-100) futures backtest data.
Built with **PyTorch**, **FAISS**, **sentence-transformers**, and **Hugging Face Transformers**.

## What It Does

Ingest CSV backtest records (strategy performance metrics), embed them with PyTorch,
store in a FAISS vector index, and answer natural-language questions about strategy
performance using a pre-trained LLM with cited sources.

## RAG Workflow (How Data Flows Through the System)

```
CSV → Python creates chunks + metadata
    ↓
PyTorch embeds TEXT chunks into vectors (not metadata)
    ↓
FAISS stores vectors + metadata together
    ↓
User asks question → PyTorch embeds question into vector
    ↓
FAISS searches vectors (hybrid: dense meaning + sparse keywords)
    ↓
Python filters by metadata (strategy, years, etc.)
    ↓
Matching TEXT chunks are handed to the LLM as context
    ↓
LLM reads the text chunks → writes answer
```

### Step-by-Step

| Step | What Happens | Tool |
|------|-------------|------|
| 1 | CSV backtest records loaded | Python (`ingest_documents.py`) |
| 2 | Each record split into text chunks + metadata labels | Python |
| 3 | Text chunks converted to 384-dim vectors | PyTorch (`sentence-transformers`) |
| 4 | Vectors + metadata stored in searchable index | FAISS |
| 5 | User question converted to vector | PyTorch |
| 6 | FAISS finds most similar vectors (hybrid dense + sparse) | FAISS |
| 7 | Results filtered by metadata (strategy name, years, etc.) | Python |
| 8 | Retrieved text chunks injected as context into prompt | Python |
| 9 | LLM reads context and generates answer | Hugging Face `transformers` |

## Example Questions

- "Which strategy had the best Sharpe ratio in 2022-2023?"
- "Show me TPO Reversal stats for 2022-2023"
- "Which strategies work best when VIX is above 25?"
- "Compare TPO Reversal vs Breakout Pullback risk-adjusted returns"

## Project Structure

```
rag-pytorch-project/
├── data/
│   ├── sample_backtests.csv   # Raw backtest data
│   └── chunks.json            # Chunks + metadata after ingestion
├── vector_store_index/        # FAISS index (created after build_index.py)
├── ingest_documents.py        # Phase 1: CSV → chunks + metadata
├── embeddings.py              # Phase 2: PyTorch dense + TF-IDF sparse embeddings
├── vector_store.py            # Phase 3: FAISS index + search + filtering
├── rag_chain.py               # Phase 4: Retrieve + generate answers
├── build_index.py             # One-time index builder
├── app.py                     # CLI interface
├── requirements.txt           # Dependencies
└── README.md
```

## Tech Stack & Interview Talking Points

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Embeddings | PyTorch + sentence-transformers | Dense semantic vectors (384-dim) |
| Sparse | TF-IDF (scikit-learn) | Keyword-based retrieval signal |
| Vector Search | FAISS | Fast similarity search with metadata filtering |
| Generation | Hugging Face `transformers` | Pre-trained LLM reads context, answers questions |
| Data | CSV + JSON | Backtest records → text chunks |

**Key design decisions:**
- **Hybrid retrieval**: 70% dense (PyTorch) + 30% sparse (TF-IDF) for semantic + keyword matching
- **Metadata filtering**: FAISS returns candidates, Python filters by strategy/years/metrics
- **Lazy loading**: AI models download only when first used, not at import
- **Chunking**: Each backtest record split into performance + regime chunks for precise retrieval

## Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Build the index (downloads embedding model ~80MB)
```bash
python build_index.py
```

### 3. Run the CLI (downloads LLM model on first question)
```bash
python app.py
```

## Hardware Notes

- **Embedding model** (`all-MiniLM-L6-v2`): 80MB, runs fine on any laptop CPU
- **LLM**: Defaults to `gpt2` (~500MB, CPU-compatible). Swap to `TinyLlama` or `phi-2` for better quality if you have more RAM.
- No GPU required for this portfolio version.

## Resume / LinkedIn Bullets

- Built a RAG system for NQ futures backtest data using PyTorch, FAISS, and Hugging Face Transformers
- Implemented hybrid retrieval: dense embeddings (sentence-transformers) + sparse TF-IDF with metadata filtering
- Designed lazy-loading architecture so AI models download only at inference time, not import
- Created end-to-end pipeline: CSV ingestion → chunking → vector indexing → LLM-generated answers with citations
- Demonstrates PyTorch tensor operations, vector search, and production-ready RAG architecture

## License

MIT
