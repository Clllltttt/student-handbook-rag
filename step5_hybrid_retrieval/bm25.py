"""A small, transparent BM25 implementation for Chinese handbook retrieval."""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


CHINESE_RUN = re.compile(r"[\u4e00-\u9fff]+")
ASCII_TOKEN = re.compile(r"[a-zA-Z0-9]+")


def tokenize(text: str) -> list[str]:
    """Tokenize Chinese as character bi/tri-grams and keep ASCII words.

    This avoids an extra word-segmentation dependency while still preserving
    exact phrases such as “招生”“公开公平公正” and article numbers.
    """
    normalized = text.lower().replace(" ", "")
    tokens = ASCII_TOKEN.findall(normalized)
    for run in CHINESE_RUN.findall(normalized):
        if len(run) == 1:
            tokens.append(run)
            continue
        tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
        if len(run) >= 3:
            tokens.extend(run[i : i + 3] for i in range(len(run) - 2))
    return tokens


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_index(
    metadata_path: Path,
    output_path: Path,
    *,
    title_repeat: int = 3,
) -> dict:
    documents = load_jsonl(metadata_path)
    postings: dict[str, list[list[int]]] = defaultdict(list)
    doc_lengths: list[int] = []

    for doc_index, item in enumerate(documents):
        tokens = tokenize(item["content"])
        tokens.extend(tokenize(item["title"]) * title_repeat)
        frequencies = Counter(tokens)
        doc_lengths.append(len(tokens))
        for term, frequency in frequencies.items():
            postings[term].append([doc_index, frequency])

    index = {
        "version": 1,
        "tokenizer": "chinese_character_bigrams_and_trigrams_plus_ascii",
        "title_repeat": title_repeat,
        "document_count": len(documents),
        "average_document_length": sum(doc_lengths) / len(doc_lengths),
        "document_lengths": doc_lengths,
        "chunk_ids": [item["chunk_id"] for item in documents],
        "postings": dict(postings),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(index, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    return index


class BM25Search:
    def __init__(
        self,
        index_path: Path,
        metadata_path: Path,
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.index = json.loads(index_path.read_text(encoding="utf-8"))
        self.metadata = load_jsonl(metadata_path)
        self.k1 = k1
        self.b = b
        chunk_ids = [item["chunk_id"] for item in self.metadata]
        if chunk_ids != self.index["chunk_ids"]:
            raise ValueError("BM25 index and metadata are out of sync")

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        query_terms = Counter(tokenize(query))
        scores: dict[int, float] = defaultdict(float)
        n_docs = self.index["document_count"]
        avgdl = self.index["average_document_length"]
        lengths = self.index["document_lengths"]

        for term, query_frequency in query_terms.items():
            posting = self.index["postings"].get(term)
            if not posting:
                continue
            document_frequency = len(posting)
            idf = math.log(
                1 + (n_docs - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            for doc_index, term_frequency in posting:
                denominator = term_frequency + self.k1 * (
                    1 - self.b + self.b * lengths[doc_index] / avgdl
                )
                scores[doc_index] += (
                    idf
                    * term_frequency
                    * (self.k1 + 1)
                    / denominator
                    * query_frequency
                )

        ordered = sorted(scores, key=lambda i: (-scores[i], i))[:top_k]
        return [
            {
                "rank": rank,
                "score": float(scores[doc_index]),
                **self.metadata[doc_index],
            }
            for rank, doc_index in enumerate(ordered, start=1)
        ]

