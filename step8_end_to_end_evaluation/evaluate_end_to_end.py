"""Run and diagnose the full handbook RAG pipeline on all 42 cases."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "step7_generation"))

from answer_question import create_searcher, source_record  # noqa: E402
from deepseek_client import DeepSeekAnswerClient  # noqa: E402
from prompts import SYSTEM_PROMPT, build_user_prompt  # noqa: E402
from judge_client import DeepSeekJudge  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def has_gold(case: dict, sources: list[dict]) -> bool | None:
    if not case["answerable"]:
        return None
    return any(
        source["title"] == case["expected_title"]
        and case["expected_text"] in source["content"]
        for source in sources
    )


def rate(rows: list[dict], predicate) -> float:
    return sum(bool(predicate(row)) for row in rows) / len(rows) if rows else 0.0


def classify_failure(row: dict) -> str | None:
    if row.get("error"):
        return "api_or_validation_error"
    if row["answerable"] and not row["candidate_has_gold"]:
        return "retrieval_error"
    if row["answerable"] != row["predicted_answerable"]:
        return "answerability_error"
    if row["answerable"] and not row["cited_has_gold"]:
        return "citation_selection_error"
    judge = row["judge"]
    if not judge["answer_correct"]:
        return "answer_correctness_error"
    if not judge["faithful"]:
        return "faithfulness_error"
    if not judge["citation_correct"]:
        return "citation_error"
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases",
        type=Path,
        default=PROJECT_ROOT / "step4_evaluation" / "test_cases.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    load_dotenv(PROJECT_ROOT / ".env", override=False)
    cases = load_jsonl(args.cases)
    if args.limit is not None:
        cases = cases[: args.limit]

    searcher = create_searcher(PROJECT_ROOT)
    generator = DeepSeekAnswerClient()
    judge_client = DeepSeekJudge()
    details = []
    run_started = time.perf_counter()

    for index, case in enumerate(cases, start=1):
        case_started = time.perf_counter()
        row = {**case, "error": None}
        try:
            retrieval_started = time.perf_counter()
            candidates = searcher.retrieve(
                case["question"], top_k=5, rerank_k=5, batch_size=5
            )
            retrieval_seconds = time.perf_counter() - retrieval_started
            evidence = [
                source_record(item, evidence_id)
                for evidence_id, item in enumerate(candidates, start=1)
            ]

            generation_started = time.perf_counter()
            generated = generator.generate(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=build_user_prompt(case["question"], candidates),
                evidence_count=len(candidates),
            )
            generation_seconds = time.perf_counter() - generation_started
            cited_sources = [
                evidence[evidence_id - 1]
                for evidence_id in generated.citation_ids
            ]

            judge_started = time.perf_counter()
            judged = judge_client.evaluate(
                case=case,
                answer=generated.answer,
                predicted_answerable=generated.answerable,
                cited_sources=cited_sources,
            )
            judge_seconds = time.perf_counter() - judge_started

            row.update(
                {
                    "predicted_answerable": generated.answerable,
                    "answer": generated.answer,
                    "reason": generated.reason,
                    "citation_ids": generated.citation_ids,
                    "candidate_has_gold": has_gold(case, evidence),
                    "cited_has_gold": has_gold(case, cited_sources),
                    "candidate_sources": [
                        {
                            key: source[key]
                            for key in (
                                "evidence_id",
                                "chunk_id",
                                "title",
                                "printed_page_start",
                                "printed_page_end",
                                "reranker_score",
                            )
                        }
                        for source in evidence
                    ],
                    "cited_sources": cited_sources,
                    "judge": asdict(judged),
                    "usage": {
                        "generation_prompt_tokens": generated.prompt_tokens,
                        "generation_completion_tokens": generated.completion_tokens,
                        "judge_prompt_tokens": judged.prompt_tokens,
                        "judge_completion_tokens": judged.completion_tokens,
                        "generation_api_calls": generated.api_calls,
                        "judge_api_calls": judged.api_calls,
                    },
                    "latency_seconds": {
                        "retrieval": round(retrieval_seconds, 3),
                        "generation": round(generation_seconds, 3),
                        "judge": round(judge_seconds, 3),
                        "total": round(time.perf_counter() - case_started, 3),
                    },
                }
            )
        except Exception as exc:  # Continue the batch and preserve the failing case.
            row["error"] = f"{type(exc).__name__}: {exc}"

        row["failure_type"] = classify_failure(row) if not row["error"] else "api_or_validation_error"
        details.append(row)
        elapsed = time.perf_counter() - run_started
        print(
            f"[{index:02d}/{len(cases)}] {case['id']} "
            f"predicted={row.get('predicted_answerable')} "
            f"failure={row['failure_type']} elapsed={elapsed:.1f}s",
            flush=True,
        )

    completed = [row for row in details if not row["error"]]
    positives = [row for row in completed if row["answerable"]]
    negatives = [row for row in completed if not row["answerable"]]
    latencies = [row["latency_seconds"]["total"] for row in completed]

    def token_total(name: str) -> int:
        return sum(row["usage"].get(name) or 0 for row in completed)

    metrics = {
        "cases": len(cases),
        "completed_cases": len(completed),
        "errors": len(cases) - len(completed),
        "answerability_accuracy": rate(
            completed,
            lambda row: row["answerable"] == row["predicted_answerable"],
        ),
        "positive_answer_rate": rate(
            positives, lambda row: row["predicted_answerable"]
        ),
        "negative_rejection_rate": rate(
            negatives, lambda row: not row["predicted_answerable"]
        ),
        "candidate_gold_recall_at_5": rate(
            positives, lambda row: row["candidate_has_gold"]
        ),
        "gold_citation_rate": rate(
            positives, lambda row: row["cited_has_gold"]
        ),
        "judge_answer_correct_rate": rate(
            completed, lambda row: row["judge"]["answer_correct"]
        ),
        "judge_faithfulness_rate": rate(
            completed, lambda row: row["judge"]["faithful"]
        ),
        "judge_citation_correct_rate": rate(
            completed, lambda row: row["judge"]["citation_correct"]
        ),
        "strict_end_to_end_pass_rate": rate(
            completed, lambda row: row["failure_type"] is None
        ),
        "latency_seconds": {
            "mean": round(statistics.mean(latencies), 3) if latencies else None,
            "median": round(statistics.median(latencies), 3) if latencies else None,
            "p95": round(
                sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)], 3
            )
            if latencies
            else None,
            "total": round(time.perf_counter() - run_started, 3),
        },
        "token_usage": {
            "generation_prompt": token_total("generation_prompt_tokens"),
            "generation_completion": token_total("generation_completion_tokens"),
            "judge_prompt": token_total("judge_prompt_tokens"),
            "judge_completion": token_total("judge_completion_tokens"),
            "generation_api_calls": token_total("generation_api_calls"),
            "judge_api_calls": token_total("judge_api_calls"),
        },
    }
    report = {
        "configuration": {
            "generator_model": generator.model,
            "judge_model": judge_client.model,
            "retrieval_top_k": 5,
            "judge_type": "same-model LLM judge plus deterministic evidence checks",
        },
        "metrics": metrics,
        "details": details,
    }
    (args.output_dir / "evaluation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    failures = [row for row in completed if row["failure_type"]]
    lines = [
        "# 第八步端到端评估报告",
        "",
        f"评测题数：{len(cases)}；成功完成：{len(completed)}；调用错误：{metrics['errors']}。",
        "",
        "## 核心指标",
        "",
        f'- 可回答性判断准确率：{metrics["answerability_accuracy"]:.1%}',
        f'- 有答案题回答率：{metrics["positive_answer_rate"]:.1%}',
        f'- 无答案题拒答率：{metrics["negative_rejection_rate"]:.1%}',
        f'- 正确证据 Recall@5：{metrics["candidate_gold_recall_at_5"]:.1%}',
        f'- 正确证据引用率：{metrics["gold_citation_rate"]:.1%}',
        f'- 评审答案正确率：{metrics["judge_answer_correct_rate"]:.1%}',
        f'- 评审忠实度：{metrics["judge_faithfulness_rate"]:.1%}',
        f'- 评审引用正确率：{metrics["judge_citation_correct_rate"]:.1%}',
        f'- 严格端到端通过率：{metrics["strict_end_to_end_pass_rate"]:.1%}',
        "",
        "## 性能与用量",
        "",
        f'- 平均每题：{metrics["latency_seconds"]["mean"]} 秒',
        f'- 中位数：{metrics["latency_seconds"]["median"]} 秒',
        f'- P95：{metrics["latency_seconds"]["p95"]} 秒',
        f'- 总耗时：{metrics["latency_seconds"]["total"]} 秒',
        f'- 生成输入/输出 token：{metrics["token_usage"]["generation_prompt"]} / {metrics["token_usage"]["generation_completion"]}',
        f'- 评审输入/输出 token：{metrics["token_usage"]["judge_prompt"]} / {metrics["token_usage"]["judge_completion"]}',
        f'- 生成/评审 API 请求次数：{metrics["token_usage"]["generation_api_calls"]} / {metrics["token_usage"]["judge_api_calls"]}',
        "",
        "## 失败项",
        "",
    ]
    if failures:
        for row in failures:
            lines.append(
                f'- {row["id"]} `{row["failure_type"]}`：{row["question"]}；'
                f'答案：{row["answer"]}；评审：{row["judge"]["notes"]}'
            )
    else:
        lines.append("无。")
    lines.extend(
        [
            "",
            "## 方法限制",
            "",
            "答案正确性、忠实度和引用质量由同一 DeepSeek 模型在独立请求中评审，可能存在偏差；正确证据召回和引用则使用人工标注的制度标题与原文片段做确定性检查。上线前仍应由人工抽查高风险问题。",
        ]
    )
    (args.output_dir / "evaluation_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
