"""Evaluate dense retrieval with Recall@K and reciprocal rank."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "step3_embedding"))

from search import VectorSearch  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cases = load_jsonl(args.cases)
    searcher = VectorSearch(args.index_dir, args.model_dir)
    details = []

    for case in cases:
        results = searcher.retrieve(case["question"], args.top_k)
        matching_rank = None
        if case["answerable"]:
            for item in results:
                if (
                    item["title"] == case["expected_title"]
                    and case["expected_text"] in item["content"]
                ):
                    matching_rank = item["rank"]
                    break
        details.append(
            {
                **case,
                "matching_rank": matching_rank,
                "top_score": round(results[0]["score"], 6),
                "top_title": results[0]["title"],
                "top_chunk_id": results[0]["chunk_id"],
                "results": [
                    {
                        "rank": item["rank"],
                        "score": round(item["score"], 6),
                        "chunk_id": item["chunk_id"],
                        "title": item["title"],
                        "pages": [
                            item["printed_page_start"],
                            item["printed_page_end"],
                        ],
                    }
                    for item in results
                ],
            }
        )

    positives = [item for item in details if item["answerable"]]
    negatives = [item for item in details if not item["answerable"]]

    def recall_at(k: int, rows: list[dict]) -> float:
        return sum(
            item["matching_rank"] is not None and item["matching_rank"] <= k
            for item in rows
        ) / len(rows)

    category_rows: dict[str, list[dict]] = defaultdict(list)
    for item in positives:
        category_rows[item["category"]].append(item)

    metrics = {
        "model": "BAAI/bge-small-zh-v1.5",
        "positive_cases": len(positives),
        "negative_diagnostics": len(negatives),
        "recall_at_1": recall_at(1, positives),
        "recall_at_3": recall_at(3, positives),
        "recall_at_5": recall_at(5, positives),
        "recall_at_10": recall_at(10, positives),
        "mrr_at_10": sum(
            1 / item["matching_rank"] if item["matching_rank"] else 0
            for item in positives
        )
        / len(positives),
        "category_recall_at_5": {
            category: recall_at(5, rows) for category, rows in category_rows.items()
        },
    }
    report = {"metrics": metrics, "details": details}
    (args.output_dir / "evaluation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    failures = [
        item
        for item in positives
        if item["matching_rank"] is None or item["matching_rank"] > 5
    ]
    markdown = [
        "# 第四步检索评估报告",
        "",
        f'- 正样本：{len(positives)}',
        f'- 无答案诊断题：{len(negatives)}',
        f'- Recall@1：{metrics["recall_at_1"]:.1%}',
        f'- Recall@3：{metrics["recall_at_3"]:.1%}',
        f'- Recall@5：{metrics["recall_at_5"]:.1%}',
        f'- Recall@10：{metrics["recall_at_10"]:.1%}',
        f'- MRR@10：{metrics["mrr_at_10"]:.3f}',
        "",
        "## Recall@5 失败项",
        "",
    ]
    if failures:
        for item in failures:
            markdown.append(
                f'- {item["id"]}：{item["question"]}；'
                f'正确结果排名：{item["matching_rank"]}'
            )
    else:
        markdown.append("无。")

    markdown.extend(["", "## 无答案问题的最高分结果", ""])
    for item in negatives:
        markdown.append(
            f'- {item["id"]}：{item["question"]} -> '
            f'{item["top_title"]}（{item["top_score"]:.4f}）'
        )
    markdown.extend(
        [
            "",
            "注意：无答案题不计入 Recall。Embedding 相似度不能直接证明资料中存在答案，后续仍需回答阶段进行证据判断。",
        ]
    )
    (args.output_dir / "evaluation_report.md").write_text(
        "\n".join(markdown), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
