"""Validate that every positive test case has ground-truth evidence in the index."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python validate_testset.py TEST_CASES_JSONL METADATA_JSONL")
    cases = load_jsonl(Path(sys.argv[1]))
    metadata = load_jsonl(Path(sys.argv[2]))

    ids = [case["id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate test case IDs")

    missing = []
    for case in cases:
        if not case["answerable"]:
            continue
        matches = [
            row
            for row in metadata
            if row["title"] == case["expected_title"]
            and case["expected_text"] in row["content"]
        ]
        if not matches:
            missing.append(case)

    if missing:
        for case in missing:
            print(
                f'MISSING {case["id"]}: {case["expected_title"]} / '
                f'{case["expected_text"]}'
            )
        raise SystemExit(f"{len(missing)} positive cases have no ground-truth chunk")

    positives = sum(case["answerable"] for case in cases)
    negatives = len(cases) - positives
    print(f"validated={len(cases)}, positives={positives}, negatives={negatives}")


if __name__ == "__main__":
    main()
