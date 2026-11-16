"""Inspect chunks containing a keyword before building embeddings.

Usage:
    python inspect_chunks.py handbook_chunks.jsonl 住宿费
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python inspect_chunks.py CHUNKS_JSONL KEYWORD")

    path = Path(sys.argv[1])
    keyword = sys.argv[2]
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    matches = [row for row in rows if keyword in row["content"]]

    print(f"keyword={keyword!r}, matches={len(matches)}")
    for row in matches[:10]:
        print("\n" + "=" * 72)
        print(
            f'{row["chunk_id"]} | {row["title"]} | '
            f'第{row["printed_page_start"]}-{row["printed_page_end"]}页 | '
            f'{row["character_count"]}字'
        )
        print(row["content"])


if __name__ == "__main__":
    main()
