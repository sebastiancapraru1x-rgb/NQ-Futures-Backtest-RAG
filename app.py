"""
app.py — Simple CLI to test the RAG system.
Run: python app.py
Model downloads happen only when you ask your first question.
"""

from rag_chain import RAGChain
from pathlib import Path


def main():
    base = Path(__file__).parent
    vs_path = base / "vector_store_index"   # Created after first build

    rag = RAGChain(vector_store_path=str(vs_path))

    # Load vector store if it exists
    if vs_path.exists():
        rag.load_vector_store(str(vs_path))
    else:
        print(f"[App] Vector store not found at {vs_path}")
        print("[App] Run build_index.py first to create it.")
        return

    print("\n" + "=" * 60)
    print("NQ Futures Backtest RAG — Ask me anything")
    print("Type 'quit' to exit")
    print("=" * 60 + "\n")

    while True:
        try:
            query = input("You: ").strip()
            if query.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break
            if not query:
                continue

            print("\nThinking...\n")
            answer = rag.generate(query, top_k=5)
            print(f"RAG: {answer}\n")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()