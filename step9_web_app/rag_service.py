"""Reusable RAG service for the local web application."""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_FRONT_MATTER_OFFSET = 6

sys.path.insert(0, str(PROJECT_ROOT / "step7_generation"))

from answer_question import create_searcher, source_record  # noqa: E402
from deepseek_client import DeepSeekAnswerClient  # noqa: E402
from prompts import SYSTEM_PROMPT, build_user_prompt  # noqa: E402


class HandbookRAGService:
    """Load the local models once and serialize CPU-heavy inference calls."""

    def __init__(self, project_root: Path = PROJECT_ROOT) -> None:
        self.project_root = project_root.resolve()
        self._searcher = None
        self._client = None
        self._lock = threading.Lock()

    @property
    def model_loaded(self) -> bool:
        return self._searcher is not None

    def _ensure_ready(self) -> None:
        load_dotenv(self.project_root / ".env", override=False)
        if self._client is None:
            self._client = DeepSeekAnswerClient()
        if self._searcher is None:
            self._searcher = create_searcher(self.project_root)

    @staticmethod
    def _web_source(source: dict) -> dict:
        item = dict(source)
        item["pdf_page_start"] = (
            item["printed_page_start"] + PDF_FRONT_MATTER_OFFSET
        )
        item["pdf_page_end"] = item["printed_page_end"] + PDF_FRONT_MATTER_OFFSET
        return item

    def ask(self, question: str, top_k: int = 5) -> dict:
        started = time.perf_counter()
        with self._lock:
            self._ensure_ready()
            assert self._searcher is not None
            assert self._client is not None

            candidates = self._searcher.retrieve(
                question,
                top_k=top_k,
                rerank_k=top_k,
                batch_size=top_k,
            )
            evidence = [
                source_record(item, index)
                for index, item in enumerate(candidates, start=1)
            ]
            generated = self._client.generate(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=build_user_prompt(question, candidates),
                evidence_count=len(candidates),
            )

        cited = [evidence[evidence_id - 1] for evidence_id in generated.citation_ids]
        return {
            "question": question,
            **asdict(generated),
            "sources": [self._web_source(item) for item in cited],
            "elapsed_ms": round((time.perf_counter() - started) * 1000),
        }
