"""Convert handbook chunks into normalized dense vectors and save a local index."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from embedding_model import BGEEmbedder, MAX_TOKENS, MODEL_NAME


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit(
            "Usage: python build_index.py CHUNKS_JSONL INDEX_DIRECTORY MODEL_DIRECTORY"
        )

    chunks_path = Path(sys.argv[1])
    index_dir = Path(sys.argv[2])
    model_dir = Path(sys.argv[3])
    index_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    chunks = load_jsonl(chunks_path)
    texts = [f'{chunk["title"]}\n{chunk["content"]}' for chunk in chunks]
    embedder = BGEEmbedder(model_dir)

    lengths = embedder.token_lengths(texts)
    over_limit = [
        {"chunk_id": chunk["chunk_id"], "tokens": length}
        for chunk, length in zip(chunks, lengths)
        if length > MAX_TOKENS
    ]
    if over_limit:
        report_path = index_dir / "overlong_chunks.json"
        report_path.write_text(
            json.dumps(over_limit, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        raise ValueError(
            f"{len(over_limit)} chunks exceed {MAX_TOKENS} tokens. "
            f"See {report_path}; fix chunking before embedding."
        )

    vectors = embedder.encode(texts, is_query=False)
    np.save(index_dir / "vectors.npy", vectors)

    metadata_path = index_dir / "metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8", newline="\n") as stream:
        for chunk in chunks:
            stream.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    config = {
        "model": MODEL_NAME,
        "vector_count": int(vectors.shape[0]),
        "vector_dimension": int(vectors.shape[1]),
        "normalized": True,
        "similarity": "cosine via dot product",
        "maximum_tokens": MAX_TOKENS,
        "maximum_observed_tokens": max(lengths),
        "device": str(embedder.device),
    }
    (index_dir / "index_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(config, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
