"""
rag_chain.py — Phase 4: Retrieve relevant chunks + generate answer.
Model loading is INSIDE functions, not at import time.
You can import this file WITHOUT downloading any models.
"""

import os
from typing import List, Optional, Dict

# Import local modules
from embeddings import HybridEmbedder
from vector_store import VectorStore


class RAGChain:
    """
    Ties everything together:
    1. Load embeddings + vector store
    2. Retrieve top-k chunks for a user question
    3. Build a prompt with retrieved context
    4. Generate answer using an LLM (model loaded lazily)
    """

    def __init__(self, vector_store_path: Optional[str] = None):
        self.vector_store: Optional[VectorStore] = None
        self.hybrid_embedder: Optional[HybridEmbedder] = None
        self.llm = None          # Loaded only when generate() is called
        self.tokenizer = None    # Loaded only when generate() is called
        self.vector_store_path = vector_store_path

    # ── Step A: Load vector store (no AI model download yet) ─────────────────
    def load_vector_store(self, path: str):
        """Load FAISS index + documents from disk. No heavy model download."""
        self.vector_store = VectorStore.load(path)
        self.hybrid_embedder = HybridEmbedder()
        self.hybrid_embedder.fit_sparse(self.vector_store.documents)
        print(f"[RAG] Vector store loaded: {len(self.vector_store)} documents")

    # ── Step B: Retrieve relevant chunks ─────────────────────────────────────
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_strategy: Optional[str] = None,
        filter_years: Optional[str] = None,
    ) -> List[Dict]:
        """Find most relevant chunks using hybrid dense + sparse search."""
        if self.vector_store is None or self.hybrid_embedder is None:
            raise RuntimeError("Call load_vector_store() first")

        return self.vector_store.hybrid_search(
            query=query,
            hybrid_embedder=self.hybrid_embedder,
            top_k=top_k,
            alpha=0.7,
            filter_strategy=filter_strategy,
            filter_years=filter_years,
        )

    # ── Step C: Build prompt with retrieved context ──────────────────────────
    def build_prompt(self, query: str, retrieved: List[Dict]) -> str:
        """Inject retrieved chunks as context for the LLM."""
        context_blocks = []
        for i, item in enumerate(retrieved, 1):
            meta = item["metadata"]
            context_blocks.append(
                f"[Source {i}] Strategy: {meta.get('strategy', 'N/A')} | "
                f"Years: {meta.get('years', 'N/A')} | "
                f"Type: {meta.get('chunk_type', 'N/A')}\n"
                f"{item['document']}\n"
            )

        context = "\n".join(context_blocks)

        prompt = f"""You are a trading analyst assistant. Answer using ONLY the context below.
If the answer is not in the context, say "I don't have enough information."

Context:
{context}

Question: {query}

Answer:"""
        return prompt

    # ── Step D: Generate answer (DOWNLOADS MODEL HERE — lazy loading) ────────
    def generate(
        self,
        query: str,
        top_k: int = 5,
        filter_strategy: Optional[str] = None,
        filter_years: Optional[str] = None,
        model_name: str = "gpt2",   # Small model, runs on CPU
    ) -> str:
        """
        Full RAG: retrieve → build prompt → generate.
        The LLM model is downloaded ONLY when this function runs.
        """
        # 1. Retrieve
        retrieved = self.retrieve(query, top_k, filter_strategy, filter_years)

        # 2. Build prompt
        prompt = self.build_prompt(query, retrieved)

        # 3. Load LLM only now (this is where download happens if not cached)
        if self.llm is None or self.tokenizer is None:
            print(f"[RAG] Loading LLM: {model_name} (this may download the model)...")
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.llm = pipeline(
                "text-generation",
                model=self.tokenizer.name_or_path,
                tokenizer=self.tokenizer,
                device=-1,   # CPU = -1. Use 0 for GPU.
            )

        # 4. Generate
        result = self.llm(
            prompt,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        generated = result[0]["generated_text"]

        # Strip the prompt from the output
        answer = generated.replace(prompt, "").strip()
        return answer


if __name__ == "__main__":
    print("RAGChain module loaded. No models downloaded yet.")
    print("Run: python app.py to start the interactive CLI.")