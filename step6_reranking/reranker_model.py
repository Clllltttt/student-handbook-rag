"""Minimal BGE cross-encoder reranker using Hugging Face Transformers."""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_NAME = "BAAI/bge-reranker-base"
MAX_TOKENS = 512


class BGEReranker:
    def __init__(self, model_dir: Path, *, local_files_only: bool = False) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            cache_dir=str(model_dir),
            local_files_only=local_files_only,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            cache_dir=str(model_dir),
            trust_remote_code=False,
            local_files_only=local_files_only,
        ).to(self.device)
        self.model.eval()

    def score(
        self,
        query: str,
        candidates: list[dict],
        *,
        batch_size: int = 4,
    ) -> list[float]:
        passages = [
            f'制度标题：{item["title"]}\n原文：{item["content"]}'
            for item in candidates
        ]
        scores: list[float] = []
        for start in range(0, len(passages), batch_size):
            batch = passages[start : start + batch_size]
            encoded = self.tokenizer(
                [query] * len(batch),
                batch,
                padding=True,
                truncation="only_second",
                max_length=MAX_TOKENS,
                return_tensors="pt",
            )
            encoded = {name: value.to(self.device) for name, value in encoded.items()}
            with torch.inference_mode():
                logits = self.model(**encoded).logits.reshape(-1)
            scores.extend(float(value) for value in logits.cpu())
        return scores

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        *,
        top_k: int | None = None,
        batch_size: int = 4,
    ) -> list[dict]:
        scores = self.score(query, candidates, batch_size=batch_size)
        rescored = [
            {
                **item,
                "retrieval_rank": item["rank"],
                "retrieval_score": item["score"],
                "reranker_score": score,
            }
            for item, score in zip(candidates, scores, strict=True)
        ]
        rescored.sort(key=lambda item: (-item["reranker_score"], item["retrieval_rank"]))
        if top_k is not None:
            rescored = rescored[:top_k]
        return [
            {**item, "rank": rank, "score": item["reranker_score"]}
            for rank, item in enumerate(rescored, start=1)
        ]

