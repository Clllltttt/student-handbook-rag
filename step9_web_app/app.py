"""Local-only Flask interface for the student handbook RAG system."""

from __future__ import annotations

import logging
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file

from rag_service import HandbookRAGService


PROJECT_ROOT = Path(__file__).resolve().parent.parent
HANDBOOK_PDF = PROJECT_ROOT / "data" / "2023研究生手册.pdf"


def create_app(service: HandbookRAGService | None = None) -> Flask:
    app = Flask(__name__)
    app.json.ensure_ascii = False
    rag = service or HandbookRAGService(PROJECT_ROOT)

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/status")
    def status():
        return jsonify(
            {
                "ok": True,
                "model_loaded": rag.model_loaded,
                "handbook": "东华大学研究生手册（2023年版）",
                "document_count": 30,
                "knowledge_chunk_count": 407,
            }
        )

    @app.post("/api/ask")
    def ask():
        payload = request.get_json(silent=True) or {}
        question = str(payload.get("question", "")).strip()
        if not question:
            return jsonify({"ok": False, "error": "请输入一个问题。"}), 400
        if len(question) > 500:
            return jsonify({"ok": False, "error": "问题请控制在 500 字以内。"}), 400

        try:
            answer = rag.ask(question)
        except RuntimeError as exc:
            app.logger.warning("RAG request failed: %s", exc)
            return jsonify({"ok": False, "error": str(exc)}), 503
        except Exception:
            app.logger.exception("Unexpected RAG request failure")
            return jsonify(
                {"ok": False, "error": "系统暂时无法回答，请稍后重试。"}
            ), 500
        return jsonify({"ok": True, **answer})

    @app.get("/handbook")
    def handbook():
        return send_file(HANDBOOK_PDF, mimetype="application/pdf")

    return app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    create_app().run(host="127.0.0.1", port=7860, debug=False, threaded=False)
