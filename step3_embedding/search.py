"""Search the local vector index with cosine similarity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from embedding_model import BGEEmbedder


def load_metadata(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class VectorSearch:
    def __init__(self, index_dir: Path, model_dir: Path) -> None:
        self.vectors = np.load(index_dir / "vectors.npy")
        self.metadata = load_metadata(index_dir / "metadata.jsonl")
        if len(self.vectors) != len(self.metadata):
            raise ValueError("Vector count and metadata count do not match")
        # Search must remain usable without internet after the model is downloaded.
        self.embedder = BGEEmbedder(model_dir, local_files_only=True)

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        query_vector = self.embedder.encode([query], is_query=True)[0]
        scores = self.vectors @ query_vector
        best_indices = np.argsort(scores)[::-1][:top_k]
        return [
            {"rank": rank, "score": float(scores[index]), **self.metadata[index]}
            for rank, index in enumerate(best_indices, start=1)
        ]


def retrieve(
    query: str,
    *,
    index_dir: Path,
    model_dir: Path,
    top_k: int,
) -> list[dict]:
    return VectorSearch(index_dir, model_dir).retrieve(query, top_k)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    results = retrieve(
        args.query,
        index_dir=args.index_dir,
        model_dir=args.model_dir,
        top_k=args.top_k,
    )
    print(f"问题：{args.query}")
    for item in results:
        print("\n" + "=" * 78)
        print(
            f'#{item["rank"]}  score={item["score"]:.4f}  '
            f'{item["chunk_id"]}'
        )
        print(
            f'{item["title"]} | '
            f'第{item["printed_page_start"]}-{item["printed_page_end"]}页'
        )
        print(item["content"])


if __name__ == "__main__":
    main()
