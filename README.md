# NQ Futures Backtest RAG

A Retrieval-Augmented Generation (RAG) system for NQ (Nasdaq-100) futures backtest data. Built with PyTorch, FAISS, sentence-transformers, and Hugging Face Transformers.

## What It Does

Ingests CSV backtest records (strategy performance metrics), embeds them with PyTorch, stores them in a FAISS vector index, and answers natural-language questions about strategy performance using a pre-trained LLM with cited sources.

## Tech Stack

- Embeddings: PyTorch + sentence-transformers
- Sparse retrieval: TF-IDF (scikit-learn)
- Vector search: FAISS
- Generation: Hugging Face Transformers
- Data: CSV + JSON

## Setup & Run

1. pip install -r requirements.txt
2. python build_index.py
3. python app.py

## License

MIT
