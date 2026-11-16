"""Extract the complete 2023 graduate handbook into one JSONL record per policy.

Usage:
    python extract_full_handbook.py INPUT_PDF OUTPUT_DIRECTORY
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import unicodedata
from pathlib import Path


PDF_TO_PRINTED_OFFSET = 6

# The handbook table of contents is the boundary source of truth.
DOCUMENTS = [
    ("国家法律法规", "中华人民共和国高等教育法", 1),
    ("国家法律法规", "普通高等学校学生管理规定", 13),
    ("国家法律法规", "高等学校学生行为准则", 29),
    ("国家法律法规", "学生伤害事故处理办法", 31),
    ("国家法律法规", "国家教育考试违规处理办法", 39),
    ("国家法律法规", "普通高等学校招生违规行为处理暂行办法", 49),
    ("国家法律法规", "高等学校预防与处理学术不端行为办法", 55),
    ("国家法律法规", "中华人民共和国网络安全法", 63),
    ("学校章程与培养管理", "东华大学章程", 79),
    ("学校章程与培养管理", "东华大学研究生学籍管理规定", 98),
    ("学校章程与培养管理", "东华大学关于进一步加强学风建设的实施细则", 110),
    ("学校章程与培养管理", "东华大学研究生参加社会实践的办法", 114),
    ("学校章程与培养管理", "东华大学关于促进毕业生就业工作的实施意见", 118),
    ("学校章程与培养管理", "东华大学引导和鼓励毕业生面向基层就业的实施办法", 124),
    ("学校章程与培养管理", "东华大学关于促进学生到国际组织实习工作的实施意见", 127),
    ("奖助与学生管理", "东华大学研究生学业奖学金评审办法", 131),
    ("奖助与学生管理", "东华大学研究生国家奖学金评审办法", 134),
    ("奖助与学生管理", "东华大学优秀研究生奖学金评审和荣誉称号授予办法", 138),
    ("奖助与学生管理", "东华大学学生违纪处分规定", 143),
    ("奖助与学生管理", "东华大学学生申诉管理规定", 155),
    ("奖助与学生管理", "东华大学博士研究生助研津贴管理办法", 159),
    ("奖助与学生管理", "东华大学研究生国家助学金管理办法", 161),
    ("奖助与学生管理", "东华大学学生帮困助学管理办法", 163),
    ("奖助与学生管理", "东华大学研究生参加“助教、助管”工作管理办法", 173),
    ("校园生活与安全", "东华大学学生社团建设管理办法", 177),
    ("校园生活与安全", "东华大学实验室安全管理办法", 185),
    ("校园生活与安全", "东华大学实验室安全检查管理办法", 197),
    ("校园生活与安全", "东华大学学生宿舍（公寓）管理办法（修订）", 210),
    ("校园生活与安全", "东华大学网络行为“十不”规范", 215),
    ("校园生活与安全", "东华大学学生补充商业医疗保险介绍", 216),
]


def compact(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    value = value.replace("（", "(").replace("）", ")")
    value = value.replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", "", value)


def remove_cjk_spacing(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    chars = r"\u3400-\u4dbf\u4e00-\u9fff，。；：、！？（）《》“”‘’"
    previous = None
    while previous != value:
        previous = value
        value = re.sub(rf"(?<=[{chars}])\s+(?=[{chars}])", "", value)
    value = re.sub(r"\s+([，。；：、！？）])", r"\1", value)
    value = re.sub(r"（\s+", "（", value)
    return re.sub(r"[ \t]+", " ", value).strip()


def clean_page(raw_page: str, printed_page: int, title: str) -> str:
    title_key = compact(title)
    lines: list[str] = []
    for raw_line in raw_page.splitlines():
        line = remove_cjk_spacing(raw_line)
        if not line:
            lines.append("")
            continue
        line_key = compact(line)
        if line == str(printed_page):
            continue
        if line_key == "研究生手册(2023年)":
            continue
        if line_key == title_key:
            continue
        lines.append(line)

    paragraphs: list[str] = []
    current = ""
    starts_item = re.compile(
        r"^(?:第[一二三四五六七八九十百]+[章节条款]|"
        r"[一二三四五六七八九十]+、|"
        r"\([一二三四五六七八九十]+\)|"
        r"\d+[、.])"
    )

    for line in lines + [""]:
        if not line:
            if current and re.search(r"[。！？；:]$", current):
                paragraphs.append(current)
                current = ""
            continue
        if current and starts_item.match(line):
            # Preserve an inline enumeration split by PDF extraction, e.g. 2、7、8月.
            if current.endswith("2、") and line == "7、":
                current += line
                continue
            paragraphs.append(current)
            current = line
        else:
            current += line

    # A PDF page often ends halfway through a sentence. Keep that fragment;
    # the next page will provide the continuation and later chunking can join it.
    if current:
        paragraphs.append(current)

    return "\n\n".join(part for part in paragraphs if part)


def extract(input_pdf: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp_dir:
        raw_path = Path(temp_dir) / "handbook.txt"
        subprocess.run(
            ["pdftotext", "-enc", "UTF-8", str(input_pdf), str(raw_path)],
            check=True,
        )
        raw_pages = raw_path.read_text(encoding="utf-8").split("\f")

    # The last split can be empty; the PDF itself has 228 physical pages.
    if raw_pages and not raw_pages[-1].strip():
        raw_pages.pop()
    if len(raw_pages) != 228:
        raise ValueError(f"Expected 228 PDF pages, extracted {len(raw_pages)}")

    records = []
    for index, (category, title, printed_start) in enumerate(DOCUMENTS):
        printed_end = (
            DOCUMENTS[index + 1][2] - 1
            if index + 1 < len(DOCUMENTS)
            else len(raw_pages) - PDF_TO_PRINTED_OFFSET
        )
        pdf_start = printed_start + PDF_TO_PRINTED_OFFSET
        pdf_end = printed_end + PDF_TO_PRINTED_OFFSET
        pages = []
        for printed_page in range(printed_start, printed_end + 1):
            pdf_page = printed_page + PDF_TO_PRINTED_OFFSET
            pages.append(
                {
                    "printed_page": printed_page,
                    "pdf_page": pdf_page,
                    "text": clean_page(raw_pages[pdf_page - 1], printed_page, title),
                }
            )

        records.append(
            {
                "document_id": f"handbook-2023-{index + 1:02d}",
                "title": title,
                "category": category,
                "handbook": "东华大学研究生手册（2023年版）",
                "source_file": str(input_pdf.resolve()),
                "printed_page_range": [printed_start, printed_end],
                "pdf_page_range": [pdf_start, pdf_end],
                "pages": pages,
                "text": "\n\n".join(page["text"] for page in pages if page["text"]),
            }
        )

    jsonl_path = output_dir / "handbook_documents.jsonl"
    with jsonl_path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    manifest = {
        "source_file": str(input_pdf.resolve()),
        "pdf_pages": len(raw_pages),
        "document_count": len(records),
        "documents": [
            {
                "document_id": item["document_id"],
                "title": item["title"],
                "category": item["category"],
                "printed_page_range": item["printed_page_range"],
                "character_count": len(item["text"]),
            }
            for item in records
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python extract_full_handbook.py INPUT_PDF OUTPUT_DIRECTORY")
    extract(Path(sys.argv[1]), Path(sys.argv[2]))


if __name__ == "__main__":
    main()
