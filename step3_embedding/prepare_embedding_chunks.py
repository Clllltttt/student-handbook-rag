"""Adapt semantic chunks to the embedding model's token window without truncation."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from transformers import AutoTokenizer

from embedding_model import MAX_TOKENS, MODEL_NAME


# Leave a small margin below the model's hard 512-token limit.
TARGET_MAX_TOKENS = 500
OVERLAP_MAX_TOKENS = 60


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def split_sentences(text: str) -> list[str]:
    return [
        part.strip()
        for part in re.split(r"(?<=[。！？；])", text)
        if part.strip()
    ]


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit(
            "Usage: python prepare_embedding_chunks.py INPUT_JSONL OUTPUT_JSONL MODEL_DIR"
        )

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    model_dir = Path(sys.argv[3])
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=str(model_dir))

    def token_count(title: str, content: str) -> int:
        return len(
            tokenizer.encode(
                f"{title}\n{content}", add_special_tokens=True, truncation=False
            )
        )

    def token_fallback(title: str, sentence: str) -> list[str]:
        title_tokens = len(tokenizer.encode(title, add_special_tokens=False)) + 3
        available = max(64, TARGET_MAX_TOKENS - title_tokens)
        ids = tokenizer.encode(sentence, add_special_tokens=False)
        parts = []
        start = 0
        while start < len(ids):
            end = min(start + available, len(ids))
            parts.append(tokenizer.decode(ids[start:end], skip_special_tokens=True))
            if end == len(ids):
                break
            start = end - OVERLAP_MAX_TOKENS
        return parts

    output_rows: list[dict] = []
    split_parent_count = 0
    for chunk in load_jsonl(input_path):
        title = chunk["title"]
        content = chunk["content"]
        if token_count(title, content) <= TARGET_MAX_TOKENS:
            parts = [content]
        else:
            split_parent_count += 1
            parts = []
            current: list[str] = []
            for sentence in split_sentences(content):
                if token_count(title, sentence) > TARGET_MAX_TOKENS:
                    if current:
                        parts.append("".join(current))
                        current = []
                    parts.extend(token_fallback(title, sentence))
                    continue

                candidate = "".join(current + [sentence])
                if current and token_count(title, candidate) > TARGET_MAX_TOKENS:
                    parts.append("".join(current))
                    previous_tail = current[-1]
                    if token_count(title, previous_tail) <= OVERLAP_MAX_TOKENS:
                        current = [previous_tail, sentence]
                        if token_count(title, "".join(current)) > TARGET_MAX_TOKENS:
                            current = [sentence]
                    else:
                        current = [sentence]
                else:
                    current.append(sentence)
            if current:
                parts.append("".join(current))

        for part_index, part in enumerate(parts, start=1):
            row = dict(chunk)
            row["parent_chunk_id"] = chunk["chunk_id"]
            if len(parts) > 1:
                row["chunk_id"] = f'{chunk["chunk_id"]}-part-{part_index:02d}'
            row["content"] = part
            row["character_count"] = len(part)
            row["embedding_token_count"] = token_count(title, part)
            if row["embedding_token_count"] > MAX_TOKENS:
                raise ValueError(f'Chunk still exceeds model limit: {row["chunk_id"]}')
            output_rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in output_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "model": MODEL_NAME,
        "input_chunks": len(load_jsonl(input_path)),
        "parents_requiring_split": split_parent_count,
        "output_chunks": len(output_rows),
        "target_max_tokens": TARGET_MAX_TOKENS,
        "hard_model_limit": MAX_TOKENS,
        "maximum_observed_tokens": max(
            row["embedding_token_count"] for row in output_rows
        ),
    }
    (output_path.parent / "embedding_chunk_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
