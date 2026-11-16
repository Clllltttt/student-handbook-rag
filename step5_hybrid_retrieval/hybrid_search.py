"""Dense + BM25 hybrid retrieval using Reciprocal Rank Fusion (RRF)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bm25 import BM25Search


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "step3_embedding"))

from search import VectorSearch  # noqa: E402


def reciprocal_rank_fusion(
    dense_results: list[dict],
    bm25_results: list[dict],
    *,
    top_k: int,
    rrf_k: int = 60,
    dense_weight: float = 1.0,
    bm25_weight: float = 1.0,
) -> list[dict]:
    """Fuse rankings, not raw scores, because the score scales differ."""
    candidates: dict[str, dict] = {}

    for channel, results, weight in (
        ("dense", dense_results, dense_weight),
        ("bm25", bm25_results, bm25_weight),
    ):
        for item in results:
            chunk_id = item["chunk_id"]
            if chunk_id not in candidates:
                candidates[chunk_id] = {
                    **item,
                    "rrf_score": 0.0,
                    "dense_rank": None,
                    "bm25_rank": None,
                    "dense_score": None,
                    "bm25_score": None,
                }
            candidate = candidates[chunk_id]
            candidate["rrf_score"] += weight / (rrf_k + item["rank"])
            candidate[f"{channel}_rank"] = item["rank"]
            candidate[f"{channel}_score"] = item["score"]

    ordered = sorted(
        candidates.values(),
        key=lambda item: (
            -item["rrf_score"],
            min(item["dense_rank"] or 10**9, item["bm25_rank"] or 10**9),
            item["chunk_id"],
        ),
    )[:top_k]
    # Put the newly assigned fused rank after the source item. Otherwise the
    # original channel's ``rank``/``score`` fields would overwrite it.
    return [
        {**item, "rank": rank, "score": item["rrf_score"]}
        for rank, item in enumerate(ordered, start=1)
    ]


class HybridSearch:
    def __init__(
        self,
        dense_index_dir: Path,
        model_dir: Path,
        bm25_index_path: Path,
        metadata_path: Path,
    ) -> None:
        self.dense = VectorSearch(dense_index_dir, model_dir)
        self.bm25 = BM25Search(bm25_index_path, metadata_path)

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        candidate_k: int = 60,
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        bm25_weight: float = 1.0,
    ) -> list[dict]:
        dense_results = self.dense.retrieve(query, candidate_k)
        bm25_results = self.bm25.retrieve(query, candidate_k)
        return reciprocal_rank_fusion(
            dense_results,
            bm25_results,
            top_k=top_k,
            rrf_k=rrf_k,
            dense_weight=dense_weight,
            bm25_weight=bm25_weight,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--dense-index-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--bm25-index", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=60)
    args = parser.parse_args()

    searcher = HybridSearch(
        args.dense_index_dir,
        args.model_dir,
        args.bm25_index,
        args.metadata,
    )
    results = searcher.retrieve(
        args.query,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
    )
    print(f"问题：{args.query}")
    for item in results:
        print("\n" + "=" * 78)
        print(
            f'#{item["rank"]} rrf={item["rrf_score"]:.6f} '
            f'dense_rank={item["dense_rank"]} bm25_rank={item["bm25_rank"]}'
        )
        print(
            f'{item["title"]} | 第{item["printed_page_start"]}-'
            f'{item["printed_page_end"]}页 | {item["chunk_id"]}'
        )
        print(item["content"])


if __name__ == "__main__":
    main()
