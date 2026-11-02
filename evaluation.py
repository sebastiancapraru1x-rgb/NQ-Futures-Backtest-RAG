"""evaluation.py - Active, label-free evaluation for the NQ RAG system."""

import re
from typing import List, Dict, Optional

import numpy as np

from embeddings import HybridEmbedder


ABSTENTION_PHRASES = (
    "i don't have enough information",
    "i do not have enough information",
    "not enough information",
    "i don't know",
)


class RAGEvaluator:
    """Label-free, active evaluator for a RAGChain instance."""

    def __init__(
        self,
        rag_chain=None,
        embedder: Optional[HybridEmbedder] = None,
        retrieval_threshold: float = 0.35,
        groundedness_threshold: float = 0.45,
    ):
        self.rag = rag_chain
        self.embedder = embedder
        self.retrieval_threshold = retrieval_threshold
        self.groundedness_threshold = groundedness_threshold

    def _embedder(self) -> HybridEmbedder:
        if self.embedder is None:
            self.embedder = HybridEmbedder()
        return self.embedder

    def _encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        return self._embedder().dense.encode(texts)

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [p.strip() for p in parts if p.strip()]

    def retrieval_confidence(self, query: str, retrieved: List[Dict]) -> Dict:
        docs = [r["document"] for r in retrieved]
        if not docs:
            return {"max_similarity": 0.0, "mean_similarity": 0.0, "num_retrieved": 0}

        q_vec = self._encode([query])[0]
        d_vecs = self._encode(docs)
        sims = d_vecs @ q_vec
        return {
            "max_similarity": float(np.max(sims)),
            "mean_similarity": float(np.mean(sims)),
            "num_retrieved": len(docs),
        }

    def context_dilution(self, retrieved: List[Dict]) -> Dict:
        docs = [r["document"] for r in retrieved]
        n = len(docs)
        if n < 2:
            return {"mean_pairwise_similarity": 1.0, "dilution_score": 0.0}

        vecs = self._encode(docs)
        sim_matrix = vecs @ vecs.T
        vals = [float(sim_matrix[i, j]) for i in range(n) for j in range(i + 1, n)]
        mean_sim = float(np.mean(vals))
        return {
            "mean_pairwise_similarity": mean_sim,
            "dilution_score": float(1.0 - mean_sim),
        }

    def groundedness(self, answer: str, retrieved: List[Dict]) -> Dict:
        docs = [r["document"] for r in retrieved]
        sentences = self._split_sentences(answer)
        if not sentences or not docs:
            return {
                "groundedness": 0.0,
                "sentence_scores": [],
                "unsupported_sentences": sentences,
            }

        sent_vecs = self._encode(sentences)
        chunk_vecs = self._encode(docs)
        scores: List[float] = []
        unsupported: List[str] = []
        for sent, svec in zip(sentences, sent_vecs):
            sims = chunk_vecs @ svec
            max_sim = float(np.max(sims)) if len(sims) else 0.0
            scores.append(max_sim)
            if max_sim < self.groundedness_threshold:
                unsupported.append(sent)

        return {
            "groundedness": float(np.mean(scores)) if scores else 0.0,
            "sentence_scores": scores,
            "unsupported_sentences": unsupported,
        }

    @staticmethod
    def _abstained(answer: str) -> bool:
        low = answer.lower()
        return any(p in low for p in ABSTENTION_PHRASES)

    def evaluate(
        self,
        query: str,
        top_k: int = 5,
        filter_strategy: Optional[str] = None,
        filter_years: Optional[str] = None,
    ) -> Dict:
        if self.rag is None:
            raise RuntimeError("RAGEvaluator requires a RAGChain instance")

        retrieved = self.rag.retrieve(
            query, top_k=top_k, filter_strategy=filter_strategy, filter_years=filter_years
        )
        answer = self.rag.generate(
            query, top_k=top_k, filter_strategy=filter_strategy, filter_years=filter_years
        )

        rc = self.retrieval_confidence(query, retrieved)
        cd = self.context_dilution(retrieved)
        gr = self.groundedness(answer, retrieved)

        low_confidence = rc["max_similarity"] < self.retrieval_threshold
        did_abstain = self._abstained(answer)
        abstention_correct = low_confidence == did_abstain

        passed = True
        if low_confidence:
            if not did_abstain:
                passed = False
        else:
            if gr["groundedness"] < self.groundedness_threshold:
                passed = False
            if did_abstain:
                passed = False

        return {
            "query": query,
            "retrieved": retrieved,
            "answer": answer,
            "metrics": {
                "retrieval_confidence": rc,
                "context_dilution": cd,
                "groundedness": gr,
                "abstention": {
                    "low_confidence": low_confidence,
                    "abstained": did_abstain,
                    "correct": abstention_correct,
                    "threshold": self.retrieval_threshold,
                },
            },
            "passed": passed,
        }
