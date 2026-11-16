"""Minimal BGE embedding wrapper using Hugging Face Transformers directly."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


MODEL_NAME = "BAAI/bge-small-zh-v1.5"
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："
MAX_TOKENS = 512


class BGEEmbedder:
    def __init__(self, model_dir: Path, *, local_files_only: bool = False) -> None:
        self.model_dir = model_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            cache_dir=str(model_dir),
            local_files_only=local_files_only,
        )
        self.model = AutoModel.from_pretrained(
            MODEL_NAME,
            cache_dir=str(model_dir),
            trust_remote_code=False,
            local_files_only=local_files_only,
        ).to(self.device)
        self.model.eval()

    def token_lengths(self, texts: Iterable[str]) -> list[int]:
        return [
            len(self.tokenizer.encode(text, add_special_tokens=True, truncation=False))
            for text in texts
        ]

    def encode(
        self,
        texts: list[str],
        *,
        is_query: bool,
        batch_size: int = 16,
    ) -> np.ndarray:
        if is_query:
            texts = [QUERY_INSTRUCTION + text for text in texts]

        batches: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_TOKENS,
                return_tensors="pt",
            )
            encoded = {name: value.to(self.device) for name, value in encoded.items()}
            with torch.inference_mode():
                output = self.model(**encoded)
                # BGE's official Transformers example uses the first token ([CLS]).
                vectors = output.last_hidden_state[:, 0]
                vectors = torch.nn.functional.normalize(vectors, p=2, dim=1)
            batches.append(vectors.cpu().numpy().astype("float32"))
        return np.concatenate(batches, axis=0)
