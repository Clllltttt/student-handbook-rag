"""Build the persistent BM25 index from embedding-chunk metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bm25 import build_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title-repeat", type=int, default=3)
    args = parser.parse_args()

    index = build_index(
        args.metadata,
        args.output,
        title_repeat=args.title_repeat,
    )
    summary = {
        "documents": index["document_count"],
        "terms": len(index["postings"]),
        "average_document_length": round(index["average_document_length"], 2),
        "title_repeat": index["title_repeat"],
        "output": str(args.output),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

