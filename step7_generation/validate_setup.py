"""Validate local files and DeepSeek configuration without exposing the key."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    required_files = [
        PROJECT_ROOT / "step3_embedding" / "index" / "vectors.npy",
        PROJECT_ROOT / "step3_embedding" / "index" / "metadata.jsonl",
        PROJECT_ROOT / "step5_hybrid_retrieval" / "index" / "bm25_index.json",
        PROJECT_ROOT / "step6_reranking" / "reranker_model.py",
    ]
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    checks = {
        "openai_package": importlib.util.find_spec("openai") is not None,
        "dotenv_package": importlib.util.find_spec("dotenv") is not None,
        "deepseek_api_key_configured": bool(api_key)
        and api_key != "在这里填写你的DeepSeek_API_Key",
        "required_files": all(path.exists() for path in required_files),
    }
    for name, passed in checks.items():
        print(f"{'[OK]' if passed else '[待配置]'} {name}")
    if not checks["deepseek_api_key_configured"]:
        print("提示：复制 .env.example 为 .env，然后只在本机填写 Key。")


if __name__ == "__main__":
    main()
