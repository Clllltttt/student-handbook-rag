"""Retrieve with hybrid search, then rerank the candidate set."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "step5_hybrid_retrieval"))

from hybrid_search import HybridSearch  # noqa: E402
from reranker_model import BGEReranker  # noqa: E402


class RerankedSearch:
    def __init__(
        self,
        dense_index_dir: Path,
        embedding_model_dir: Path,
        bm25_index_path: Path,
        metadata_path: Path,
        reranker_model_dir: Path,
        *,
        local_files_only: bool = True,
    ) -> None:
        self.hybrid = HybridSearch(
            dense_index_dir,
            embedding_model_dir,
            bm25_index_path,
            metadata_path,
        )
        self.reranker = BGEReranker(
            reranker_model_dir,
            local_files_only=local_files_only,
        )

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        rerank_k: int = 10,
        retrieval_candidate_k: int = 60,
        batch_size: int = 4,
    ) -> list[dict]:
        candidates = self.hybrid.retrieve(
            query,
            top_k=rerank_k,
            candidate_k=retrieval_candidate_k,
        )
        return self.reranker.rerank(
            query,
            candidates,
            top_k=top_k,
            batch_size=batch_size,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--dense-index-dir", type=Path, required=True)
    parser.add_argument("--embedding-model-dir", type=Path, required=True)
    parser.add_argument("--bm25-index", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--reranker-model-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--rerank-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    searcher = RerankedSearch(
        args.dense_index_dir,
        args.embedding_model_dir,
        args.bm25_index,
        args.metadata,
        args.reranker_model_dir,
    )
    results = searcher.retrieve(
        args.query,
        top_k=args.top_k,
        rerank_k=args.rerank_k,
        batch_size=args.batch_size,
    )
    print(f"问题：{args.query}")
    for item in results:
        print("\n" + "=" * 78)
        print(
            f'#{item["rank"]} reranker={item["reranker_score"]:.4f} '
            f'hybrid_rank={item["retrieval_rank"]}'
        )
        print(
            f'{item["title"]} | 第{item["printed_page_start"]}-'
            f'{item["printed_page_end"]}页 | {item["chunk_id"]}'
        )
        print(item["content"])


if __name__ == "__main__":
    main()

