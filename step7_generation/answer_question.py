"""End-to-end RAG: retrieve, rerank, generate, validate, and cite."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "step6_reranking"))

from deepseek_client import DeepSeekAnswerClient  # noqa: E402
from prompts import SYSTEM_PROMPT, build_user_prompt  # noqa: E402


def create_searcher(project_root: Path):
    # Importing PyTorch/Transformers is relatively slow, so defer it until
    # configuration has been checked and retrieval is actually needed.
    from rerank_search import RerankedSearch

    return RerankedSearch(
        project_root / "step3_embedding" / "index",
        project_root / "models",
        project_root / "step5_hybrid_retrieval" / "index" / "bm25_index.json",
        project_root / "step3_embedding" / "index" / "metadata.jsonl",
        project_root / "models",
    )


def source_record(item: dict, evidence_id: int) -> dict:
    return {
        "evidence_id": evidence_id,
        "chunk_id": item["chunk_id"],
        "title": item["title"],
        "printed_page_start": item["printed_page_start"],
        "printed_page_end": item["printed_page_end"],
        "reranker_score": round(item["reranker_score"], 6),
        "content": item["content"],
    }


def render_answer(answer: dict) -> str:
    lines = [answer["answer"]]
    if answer["answerable"] and answer["sources"]:
        lines.extend(["", "来源："])
        for source in answer["sources"]:
            start = source["printed_page_start"]
            end = source["printed_page_end"]
            pages = f"第{start}页" if start == end else f"第{start}-{end}页"
            lines.append(
                f'- [证据{source["evidence_id"]}] '
                f'{source["title"]}，{pages}'
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--show-evidence", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    load_dotenv(project_root / ".env", override=False)
    try:
        client = None if args.dry_run else DeepSeekAnswerClient()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    searcher = create_searcher(project_root)
    candidates = searcher.retrieve(
        args.question,
        top_k=args.top_k,
        rerank_k=args.top_k,
        batch_size=args.top_k,
    )
    evidence = [
        source_record(item, index)
        for index, item in enumerate(candidates, start=1)
    ]
    user_prompt = build_user_prompt(args.question, candidates)

    if args.dry_run:
        result = {
            "mode": "dry_run",
            "question": args.question,
            "evidence": evidence,
            "system_prompt": SYSTEM_PROMPT,
            "user_prompt": user_prompt,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    assert client is not None
    try:
        generated = client.generate(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            evidence_count=len(candidates),
        )
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    cited = [
        evidence[evidence_id - 1]
        for evidence_id in generated.citation_ids
    ]
    result = {
        "question": args.question,
        **asdict(generated),
        "sources": cited,
    }
    print(render_answer(result))
    if args.show_evidence:
        print("\n--- 调试信息 ---")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
