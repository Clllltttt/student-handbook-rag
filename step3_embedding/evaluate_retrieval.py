"""Small transparent retrieval evaluation set for the handbook."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from search import VectorSearch


TEST_CASES = [
    {
        "query": "研究生最长可以读几年？",
        "expected_title": "东华大学研究生学籍管理规定",
        "expected_text": "最长学习年限",
    },
    {
        "query": "研究生请假一周以上由谁审批？",
        "expected_title": "东华大学研究生学籍管理规定",
        "expected_text": "请假 1 周以上",
    },
    {
        "query": "宿舍费用是按几个月计算的？",
        "expected_title": "东华大学学生宿舍（公寓）管理办法（修订）",
        "expected_text": "全年按 9 个月计算",
    },
    {
        "query": "研究生国家助学金每年多少钱？",
        "expected_title": "东华大学研究生国家助学金管理办法",
        "expected_text": "元/生/年",
    },
    {
        "query": "宿舍里能不能使用电饭锅？",
        "expected_title": "东华大学学生宿舍（公寓）管理办法（修订）",
        "expected_text": "电饭锅",
    },
    {
        "query": "实验室发生安全事故后应该怎么报告？",
        "expected_title": "东华大学实验室安全管理办法",
        "expected_text": "事故",
    },
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    searcher = VectorSearch(args.index_dir, args.model_dir)
    details = []
    for case in TEST_CASES:
        results = searcher.retrieve(case["query"], args.top_k)
        matching_ranks = [
            item["rank"]
            for item in results
            if item["title"] == case["expected_title"]
            and case["expected_text"] in item["content"]
        ]
        details.append(
            {
                **case,
                "passed": bool(matching_ranks),
                "matching_rank": matching_ranks[0] if matching_ranks else None,
                "results": [
                    {
                        "rank": item["rank"],
                        "score": round(item["score"], 6),
                        "chunk_id": item["chunk_id"],
                        "title": item["title"],
                        "printed_page_start": item["printed_page_start"],
                        "printed_page_end": item["printed_page_end"],
                    }
                    for item in results
                ],
            }
        )

    passed = sum(item["passed"] for item in details)
    report = {
        "metric": f"Recall@{args.top_k}",
        "passed": passed,
        "total": len(details),
        "score": passed / len(details),
        "details": details,
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
