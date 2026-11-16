"""RAG step 2: split structured handbook documents into retrievable chunks.

Usage:
    python chunk_handbook.py INPUT_JSONL OUTPUT_DIRECTORY
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path


TARGET_CHARS = 600
MAX_CHARS = 900
MIN_CHARS = 180
OVERLAP_CHARS = 80

HEADING_ONLY = re.compile(
    r"^(?:第[一二三四五六七八九十百零〇0-9]+[编章节]|"
    r"[一二三四五六七八九十]+、[^。！？；]{0,20})$"
)
STRONG_BOUNDARY = re.compile(
    r"^(?:第[一二三四五六七八九十百零〇0-9]+[编章节条款]|"
    r"[一二三四五六七八九十]+、|"
    r"\([一二三四五六七八九十]+\)|"
    r"\d+[、.])"
)
PACK_BOUNDARY = re.compile(
    r"^(?:第[一二三四五六七八九十百零〇0-9]+[编章节]|"
    r"[一二三四五六七八九十]+、[^。！？；]{0,20})"
)
SENTENCE_END = re.compile(r"[。！？；:]$")


def load_documents(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def page_units(document: dict) -> list[dict]:
    """Create paragraph-like units and repair sentences broken across pages."""
    units: list[dict] = []
    for page in document["pages"]:
        printed_page = page["printed_page"]
        paragraphs = [part.strip() for part in page["text"].split("\n\n") if part.strip()]
        for index, paragraph in enumerate(paragraphs):
            if (
                index == 0
                and units
                and units[-1]["pages"][-1] == printed_page - 1
                and not SENTENCE_END.search(units[-1]["text"])
                and not HEADING_ONLY.match(units[-1]["text"])
                and not STRONG_BOUNDARY.match(paragraph)
            ):
                units[-1]["text"] += paragraph
                units[-1]["pages"].append(printed_page)
            else:
                units.append({"text": paragraph, "pages": [printed_page]})
    return units


def sentence_split(text: str) -> list[str]:
    parts = re.split(r"(?<=[。！？；])", text)
    return [part.strip() for part in parts if part.strip()]


def split_oversized(unit: dict) -> list[dict]:
    """Split a long semantic unit by sentences, then use bounded overlap if needed."""
    if len(unit["text"]) <= MAX_CHARS:
        return [unit]

    result: list[dict] = []
    current = ""
    for sentence in sentence_split(unit["text"]):
        if len(sentence) > MAX_CHARS:
            if current:
                result.append({"text": current, "pages": unit["pages"][:]})
                current = ""
            start = 0
            while start < len(sentence):
                end = min(start + MAX_CHARS, len(sentence))
                result.append(
                    {"text": sentence[start:end], "pages": unit["pages"][:]}
                )
                if end == len(sentence):
                    break
                start = end - OVERLAP_CHARS
            continue

        candidate = current + sentence
        if current and len(candidate) > MAX_CHARS:
            result.append({"text": current, "pages": unit["pages"][:]})
            current = sentence
        else:
            current = candidate
    if current:
        result.append({"text": current, "pages": unit["pages"][:]})
    return result


def make_chunks(document: dict) -> list[dict]:
    units: list[dict] = []
    pending_heading: dict | None = None

    for raw_unit in page_units(document):
        if HEADING_ONLY.match(raw_unit["text"]):
            if pending_heading:
                units.append(pending_heading)
            pending_heading = raw_unit
            continue
        if pending_heading:
            raw_unit = {
                "text": pending_heading["text"] + "\n" + raw_unit["text"],
                "pages": sorted(set(pending_heading["pages"] + raw_unit["pages"])),
            }
            pending_heading = None
        units.extend(split_oversized(raw_unit))
    if pending_heading:
        units.append(pending_heading)

    chunks: list[dict] = []
    current_text = ""
    current_pages: list[int] = []

    def flush() -> None:
        nonlocal current_text, current_pages
        if current_text.strip():
            chunks.append(
                {
                    "content": current_text.strip(),
                    "printed_pages": sorted(set(current_pages)),
                }
            )
        current_text = ""
        current_pages = []

    for unit in units:
        text = unit["text"].strip()
        boundary = bool(PACK_BOUNDARY.match(text))
        candidate = text if not current_text else current_text + "\n\n" + text

        if current_text and (len(candidate) > MAX_CHARS or boundary):
            flush()

        if len(text) > MAX_CHARS:
            # Defensive fallback; split_oversized should normally prevent this.
            for part in split_oversized(unit):
                chunks.append(
                    {
                        "content": part["text"],
                        "printed_pages": sorted(set(part["pages"])),
                    }
                )
            continue

        current_text = text if not current_text else current_text + "\n\n" + text
        current_pages.extend(unit["pages"])

        if len(current_text) >= TARGET_CHARS:
            flush()

    flush()

    # Fold very small fragments into a neighbour. This keeps enough context for
    # retrieval without breaking the 900-character safety limit.
    compacted: list[dict] = []
    for chunk in chunks:
        if compacted and (
            len(chunk["content"]) < MIN_CHARS
            or len(compacted[-1]["content"]) < MIN_CHARS
        ):
            combined = compacted[-1]["content"] + "\n\n" + chunk["content"]
            if len(combined) <= MAX_CHARS:
                compacted[-1]["content"] = combined
                compacted[-1]["printed_pages"] = sorted(
                    set(compacted[-1]["printed_pages"] + chunk["printed_pages"])
                )
                continue
        compacted.append(chunk)

    output = []
    for index, chunk in enumerate(compacted, start=1):
        pages = chunk["printed_pages"]
        output.append(
            {
                "chunk_id": f'{document["document_id"]}-chunk-{index:03d}',
                "document_id": document["document_id"],
                "title": document["title"],
                "category": document["category"],
                "handbook": document["handbook"],
                "printed_page_start": min(pages),
                "printed_page_end": max(pages),
                "content": chunk["content"],
                "character_count": len(chunk["content"]),
            }
        )
    return output


def build(input_path: Path, output_dir: Path) -> None:
    documents = load_documents(input_path)
    chunks = [chunk for document in documents for chunk in make_chunks(document)]
    output_dir.mkdir(parents=True, exist_ok=True)

    chunks_path = output_dir / "handbook_chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8", newline="\n") as stream:
        for chunk in chunks:
            stream.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    sizes = [chunk["character_count"] for chunk in chunks]
    report = {
        "input_document_count": len(documents),
        "chunk_count": len(chunks),
        "chunking_strategy": "structure-first, sentence fallback",
        "target_characters": TARGET_CHARS,
        "maximum_characters": MAX_CHARS,
        "minimum_preferred_characters": MIN_CHARS,
        "overlap_characters_for_oversized_sentences": OVERLAP_CHARS,
        "size_statistics": {
            "minimum": min(sizes),
            "median": statistics.median(sizes),
            "mean": round(statistics.mean(sizes), 2),
            "maximum": max(sizes),
        },
        "chunks_per_document": {
            document["document_id"]: sum(
                1 for chunk in chunks if chunk["document_id"] == document["document_id"]
            )
            for document in documents
        },
    }
    (output_dir / "chunking_stats.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python chunk_handbook.py INPUT_JSONL OUTPUT_DIRECTORY")
    build(Path(sys.argv[1]), Path(sys.argv[2]))


if __name__ == "__main__":
    main()
