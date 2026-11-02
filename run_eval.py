"""run_eval.py - CLI to evaluate the NQ RAG system (active, label-free)."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from rag_chain import RAGChain
from evaluation import RAGEvaluator


def main():
    parser = argparse.ArgumentParser(description="Evaluate the NQ RAG system.")
    parser.add_argument("--query", type=str, default=None, help="Single query to evaluate.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve.")
    parser.add_argument("--output", type=str, default="eval_results.json", help="Output JSON file.")
    args = parser.parse_args()

    base = Path(__file__).parent
    vs_path = base / "vector_store_index"
    if not vs_path.exists():
        print(f"[Eval] Vector store not found at {vs_path}. Run build_index.py first.")
        sys.exit(1)

    rag = RAGChain(vector_store_path=str(vs_path))
    rag.load_vector_store(str(vs_path))
    evaluator = RAGEvaluator(rag)

    if args.query:
        queries = [args.query]
    else:
        queries = [
            "What were the IFVG_CISD_EMA stats for 2025-2026?",
            "How did IFVG_CISD_EMA perform in 2023-2024?",
            "What is the weather in New York today?",  # out-of-domain -> should abstain
        ]

    results = []
    passed_count = 0
    for q in queries:
        print("\n" + "=" * 60)
        print(f"QUERY: {q}")
        print("=" * 60)
        result = evaluator.evaluate(q, top_k=args.top_k)
        m = result["metrics"]
        rc = m["retrieval_confidence"]
        cd = m["context_dilution"]
        gr = m["groundedness"]
        ab = m["abstention"]
        print(f"Retrieval confidence : max={rc['max_similarity']:.3f} mean={rc['mean_similarity']:.3f}")
        print(f"Context dilution     : coherence={cd['mean_pairwise_similarity']:.3f} dilution={cd['dilution_score']:.3f}")
        print(f"Groundedness         : {gr['groundedness']:.3f}")
        if gr["unsupported_sentences"]:
            print("Unsupported sentences:")
            for s in gr["unsupported_sentences"]:
                print(f"  - {s}")
        print(f"Abstention           : low_conf={ab['low_confidence']} abstained={ab['abstained']} correct={ab['correct']}")
        print(f"Answer               : {result['answer'][:300]}")
        print(f"PASSED: {result['passed']}")
        if result["passed"]:
            passed_count += 1
        results.append({
            "query": result["query"],
            "answer": result["answer"],
            "metrics": result["metrics"],
            "passed": result["passed"],
        })

    out_path = base / args.output
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print("\n" + "=" * 60)
    print(f"Eval complete: {passed_count}/{len(queries)} passed")
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
