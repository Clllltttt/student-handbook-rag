"""Compare hybrid retrieval with cross-encoder reranking."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from rerank_search import RerankedSearch


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def find_rank(case: dict, results: list[dict]) -> int | None:
    if not case["answerable"]:
        return None
    for item in results:
        if (
            item["title"] == case["expected_title"]
            and case["expected_text"] in item["content"]
        ):
            return item["rank"]
    return None


def summarize(details: list[dict], mode: str) -> dict:
    positives = [item for item in details if item["answerable"]]

    def recall_at(k: int, rows: list[dict]) -> float:
        return sum(
            item["ranks"][mode] is not None and item["ranks"][mode] <= k
            for item in rows
        ) / len(rows)

    categories: dict[str, list[dict]] = defaultdict(list)
    for item in positives:
        categories[item["category"]].append(item)
    return {
        "recall_at_1": recall_at(1, positives),
        "recall_at_3": recall_at(3, positives),
        "recall_at_5": recall_at(5, positives),
        "recall_at_10": recall_at(10, positives),
        "mrr_at_10": sum(
            1 / item["ranks"][mode]
            if item["ranks"][mode] and item["ranks"][mode] <= 10
            else 0
            for item in positives
        )
        / len(positives),
        "category_recall_at_1": {
            category: recall_at(1, rows) for category, rows in categories.items()
        },
    }


def compact(results: list[dict], *, reranked: bool) -> list[dict]:
    rows = []
    for item in results:
        row = {
            "rank": item["rank"],
            "chunk_id": item["chunk_id"],
            "title": item["title"],
            "score": round(item["score"], 6),
        }
        if reranked:
            row["hybrid_rank"] = item["retrieval_rank"]
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--dense-index-dir", type=Path, required=True)
    parser.add_argument("--embedding-model-dir", type=Path, required=True)
    parser.add_argument("--bm25-index", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--reranker-model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rerank-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cases = load_jsonl(args.cases)
    searcher = RerankedSearch(
        args.dense_index_dir,
        args.embedding_model_dir,
        args.bm25_index,
        args.metadata,
        args.reranker_model_dir,
    )
    details = []
    started = time.perf_counter()

    for index, case in enumerate(cases, start=1):
        hybrid = searcher.hybrid.retrieve(
            case["question"], top_k=args.rerank_k, candidate_k=60
        )
        reranked = searcher.reranker.rerank(
            case["question"],
            hybrid,
            top_k=args.rerank_k,
            batch_size=args.batch_size,
        )
        details.append(
            {
                **case,
                "ranks": {
                    "hybrid": find_rank(case, hybrid),
                    "reranked": find_rank(case, reranked),
                },
                "top_results": {
                    "hybrid": compact(hybrid, reranked=False),
                    "reranked": compact(reranked, reranked=True),
                },
            }
        )
        elapsed = time.perf_counter() - started
        print(
            f"[{index:02d}/{len(cases)}] {case['id']} "
            f"{details[-1]['ranks']} elapsed={elapsed:.1f}s",
            flush=True,
        )

    elapsed = time.perf_counter() - started
    metrics = {
        mode: summarize(details, mode) for mode in ("hybrid", "reranked")
    }
    report = {
        "configuration": {
            "reranker_model": "BAAI/bge-reranker-base",
            "rerank_k": args.rerank_k,
            "batch_size": args.batch_size,
            "device": str(searcher.reranker.device),
        },
        "elapsed_seconds": round(elapsed, 3),
        "average_seconds_per_question": round(elapsed / len(cases), 3),
        "metrics": metrics,
        "details": details,
    }
    (args.output_dir / "reranker_evaluation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# 第六步 Reranker 评估报告",
        "",
        f"混合检索先召回 Top {args.rerank_k}，再由 `BAAI/bge-reranker-base` 交叉编码器重排。",
        "",
        "| 模式 | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR@10 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mode in ("hybrid", "reranked"):
        item = metrics[mode]
        lines.append(
            f'| {mode} | {item["recall_at_1"]:.1%} | '
            f'{item["recall_at_3"]:.1%} | {item["recall_at_5"]:.1%} | '
            f'{item["recall_at_10"]:.1%} | {item["mrr_at_10"]:.3f} |'
        )
    lines.extend(
        [
            "",
            f"CPU 总评测时间：{elapsed:.1f} 秒；平均每题 {elapsed / len(cases):.2f} 秒。",
            "",
            "## 排名发生变化的问题",
            "",
        ]
    )
    changed = False
    for item in details:
        if item["answerable"] and item["ranks"]["hybrid"] != item["ranks"]["reranked"]:
            changed = True
            lines.append(
                f'- {item["id"]}：hybrid={item["ranks"]["hybrid"]}，'
                f'reranked={item["ranks"]["reranked"]}；{item["question"]}'
            )
    if not changed:
        lines.append("无。")
    (args.output_dir / "reranker_evaluation.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
