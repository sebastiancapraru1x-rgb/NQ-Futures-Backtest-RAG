"""
ingest_documents.py — Phase 1: Load CSV backtest data → create text chunks + metadata.
NO MODEL DOWNLOADS happen here. Pure data prep.
"""

import csv
import os
import json
from typing import List, Dict


def ingest_backtest_csv(csv_path: str) -> List[Dict]:
    """Load backtest CSV and convert numeric fields."""
    records = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["total_return_pct"] = float(row["total_return_pct"])
            row["win_rate_pct"] = float(row["win_rate_pct"])
            row["sharpe_ratio"] = float(row["sharpe_ratio"])
            row["max_drawdown_pct"] = float(row["max_drawdown_pct"])
            row["avg_r_per_trade"] = float(row["avg_r_per_trade"])
            row["total_trades"] = int(row["total_trades"])
            row["avg_contracts"] = float(row["avg_contracts"])
            records.append(row)
    print(f"[Ingest] Loaded {len(records)} backtest records")
    return records


def create_strategy_chunks(records: List[Dict]) -> tuple[List[str], List[Dict]]:
    """Split each backtest record into 2 text chunks + attach metadata."""
    chunks = []
    metadatas = []

    for rec in records:
        strategy = rec["strategy"]
        years = rec["years"]
        timeframe = rec["timeframe"]

        # Chunk 1: Core performance
        text = (
            f"{strategy} strategy backtest over {years} on {timeframe} timeframe. "
            f"Total return: {rec['total_return_pct']}%. Win rate: {rec['win_rate_pct']}%. "
            f"Sharpe ratio: {rec['sharpe_ratio']}. Max drawdown: {rec['max_drawdown_pct']}%. "
            f"Avg R per trade: {rec['avg_r_per_trade']}. Total trades: {rec['total_trades']}. "
            f"Avg contracts: {rec['avg_contracts']}."
        )
        chunks.append(text)
        metadatas.append({
            "strategy": strategy, "years": years, "timeframe": timeframe,
            "chunk_type": "core_performance",
            "total_return_pct": rec["total_return_pct"], "win_rate_pct": rec["win_rate_pct"],
            "sharpe_ratio": rec["sharpe_ratio"], "max_drawdown_pct": rec["max_drawdown_pct"],
            "avg_r_per_trade": rec["avg_r_per_trade"], "total_trades": rec["total_trades"],
            "avg_contracts": rec["avg_contracts"],
        })

        # Chunk 2: Regime analysis
        regime_text = (
            f"{strategy} ({years}) regime analysis. Best performance in: {rec['best_regime']}. "
            f"Worst performance in: {rec['worst_regime']}. Best month: {rec['best_month']}. "
            f"Worst month: {rec['worst_month']}."
        )
        chunks.append(regime_text)
        metadatas.append({
            "strategy": strategy, "years": years, "timeframe": timeframe,
            "chunk_type": "regime_analysis",
            "best_regime": rec["best_regime"], "worst_regime": rec["worst_regime"],
            "best_month": rec["best_month"], "worst_month": rec["worst_month"],
        })

    print(f"[Ingest] Created {len(chunks)} chunks from {len(records)} records")
    return chunks, metadatas


def ingest_documents(csv_path: str) -> tuple[List[str], List[Dict]]:
    """Full pipeline: load CSV → create chunks."""
    records = ingest_backtest_csv(csv_path)
    return create_strategy_chunks(records)


def save_chunks(chunks: List[str], metadatas: List[Dict], output_path: str):
    """Save chunks to JSON for inspection."""
    data = [{"text": c, "metadata": m} for c, m in zip(chunks, metadatas)]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[Ingest] Saved {len(data)} chunks to {output_path}")


if __name__ == "__main__":
    csv_path = os.path.join(os.path.dirname(__file__), "data", "sample_backtests.csv")
    chunks, metadatas = ingest_documents(csv_path)
    for i, (c, m) in enumerate(zip(chunks[:4], metadatas[:4])):
        print(f"\n--- Chunk {i+1} ---\n{c}\n{m}")
    save_chunks(chunks, metadatas, os.path.join(os.path.dirname(__file__), "data", "chunks.json"))