"""Compare dense, BM25, and hybrid retrieval on the same test set."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from hybrid_search import HybridSearch, reciprocal_rank_fusion


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def matching_rank(case: dict, results: list[dict]) -> int | None:
    if not case["answerable"]:
        return None
    for item in results:
        if (
            item["title"] == case["expected_title"]
            and case["expected_text"] in item["content"]
        ):
            return item["rank"]
    return None


def summarize(rows: list[dict], mode: str) -> dict:
    positives = [row for row in rows if row["answerable"]]

    def recall_at(k: int, selected: list[dict]) -> float:
        return sum(
            row["ranks"][mode] is not None and row["ranks"][mode] <= k
            for row in selected
        ) / len(selected)

    category_rows: dict[str, list[dict]] = defaultdict(list)
    for row in positives:
        category_rows[row["category"]].append(row)

    return {
        "recall_at_1": recall_at(1, positives),
        "recall_at_3": recall_at(3, positives),
        "recall_at_5": recall_at(5, positives),
        "recall_at_10": recall_at(10, positives),
        "mrr_at_10": sum(
            1 / row["ranks"][mode]
            if row["ranks"][mode] and row["ranks"][mode] <= 10
            else 0
            for row in positives
        )
        / len(positives),
        "category_recall_at_5": {
            category: recall_at(5, selected)
            for category, selected in category_rows.items()
        },
    }


def compact(results: list[dict]) -> list[dict]:
    return [
        {
            "rank": item["rank"],
            "chunk_id": item["chunk_id"],
            "title": item["title"],
            "score": round(item["score"], 6),
        }
        for item in results[:10]
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--dense-index-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--bm25-index", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-k", type=int, default=60)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--dense-weight", type=float, default=1.0)
    parser.add_argument("--bm25-weight", type=float, default=1.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cases = load_jsonl(args.cases)
    searcher = HybridSearch(
        args.dense_index_dir,
        args.model_dir,
        args.bm25_index,
        args.metadata,
    )
    details = []

    for index, case in enumerate(cases, start=1):
        dense_candidates = searcher.dense.retrieve(case["question"], args.candidate_k)
        bm25_candidates = searcher.bm25.retrieve(case["question"], args.candidate_k)
        hybrid_candidates = reciprocal_rank_fusion(
            dense_candidates,
            bm25_candidates,
            top_k=args.candidate_k,
            rrf_k=args.rrf_k,
            dense_weight=args.dense_weight,
            bm25_weight=args.bm25_weight,
        )
        results = {
            "dense": dense_candidates,
            "bm25": bm25_candidates,
            "hybrid": hybrid_candidates,
        }
        details.append(
            {
                **case,
                "ranks": {
                    mode: matching_rank(case, mode_results)
                    for mode, mode_results in results.items()
                },
                "top_results": {
                    mode: compact(mode_results)
                    for mode, mode_results in results.items()
                },
            }
        )
        print(f"[{index:02d}/{len(cases)}] {case['id']} {details[-1]['ranks']}")

    metrics = {
        mode: summarize(details, mode) for mode in ("dense", "bm25", "hybrid")
    }
    report = {
        "configuration": {
            "candidate_k": args.candidate_k,
            "rrf_k": args.rrf_k,
            "dense_weight": args.dense_weight,
            "bm25_weight": args.bm25_weight,
        },
        "positive_cases": sum(case["answerable"] for case in cases),
        "negative_diagnostics": sum(not case["answerable"] for case in cases),
        "metrics": metrics,
        "details": details,
    }
    (args.output_dir / "comparison_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# 第五步混合检索评估报告",
        "",
        "同一套 42 道题、同一份索引，对比纯向量、纯 BM25 和 RRF 混合检索。",
        "",
        "| 模式 | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR@10 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mode in ("dense", "bm25", "hybrid"):
        item = metrics[mode]
        lines.append(
            f'| {mode} | {item["recall_at_1"]:.1%} | '
            f'{item["recall_at_3"]:.1%} | {item["recall_at_5"]:.1%} | '
            f'{item["recall_at_10"]:.1%} | {item["mrr_at_10"]:.3f} |'
        )
    lines.extend(["", "## 逐题排名变化", ""])
    for row in details:
        if row["answerable"]:
            ranks = row["ranks"]
            lines.append(
                f'- {row["id"]}：dense={ranks["dense"]}，'
                f'BM25={ranks["bm25"]}，hybrid={ranks["hybrid"]}；'
                f'{row["question"]}'
            )
    lines.extend(
        [
            "",
            "## 说明",
            "",
            "BM25 擅长精确词语，向量检索擅长语义改写；RRF 根据名次融合，不直接相加两种量纲不同的分数。无答案题仍只作诊断，不计入 Recall。",
        ]
    )
    (args.output_dir / "comparison_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

