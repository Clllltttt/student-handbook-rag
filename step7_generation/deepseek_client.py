"""DeepSeek OpenAI-compatible client with strict response validation."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from openai import OpenAI, OpenAIError


DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-flash"


@dataclass(frozen=True)
class GeneratedAnswer:
    answerable: bool
    answer: str
    citation_ids: list[int]
    reason: str
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None
    api_calls: int


class DeepSeekAnswerClient:
    def __init__(self) -> None:
        api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        if not api_key or api_key == "在这里填写你的DeepSeek_API_Key":
            raise RuntimeError(
                "未找到 DEEPSEEK_API_KEY。请复制 .env.example 为 .env，"
                "在本机填入 Key；不要把 Key 发到聊天或写进代码。"
            )
        self.model = os.getenv("DEEPSEEK_MODEL", DEFAULT_MODEL).strip()
        base_url = os.getenv("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL).strip()
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=60.0,
            max_retries=2,
        )

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        evidence_count: int,
    ) -> GeneratedAnswer:
        total_prompt_tokens = 0
        total_completion_tokens = 0
        response = None
        content = None
        api_calls = 0
        # DeepSeek documents that JSON mode may occasionally return empty
        # content. Retry once at the application layer for this specific case.
        for api_calls in range(1, 3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    extra_body={"thinking": {"type": "disabled"}},
                    temperature=0,
                    max_tokens=800,
                    stream=False,
                )
            except OpenAIError as exc:
                raise RuntimeError(f"DeepSeek API 调用失败：{exc}") from exc
            usage = response.usage
            total_prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
            total_completion_tokens += getattr(usage, "completion_tokens", 0) or 0
            content = response.choices[0].message.content
            if content:
                break
        if response is None or not content:
            raise RuntimeError("DeepSeek 连续两次返回空内容。")
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError("DeepSeek 未返回合法 JSON。") from exc

        answerable = payload.get("answerable")
        answer = payload.get("answer")
        citation_ids = payload.get("citation_ids")
        reason = payload.get("reason", "")
        if not isinstance(answerable, bool):
            raise RuntimeError("返回字段 answerable 必须是布尔值。")
        if not isinstance(answer, str) or not answer.strip():
            raise RuntimeError("返回字段 answer 必须是非空字符串。")
        if not isinstance(reason, str):
            raise RuntimeError("返回字段 reason 必须是字符串。")
        if not isinstance(citation_ids, list) or not all(
            isinstance(value, int) for value in citation_ids
        ):
            raise RuntimeError("返回字段 citation_ids 必须是整数数组。")
        citation_ids = list(dict.fromkeys(citation_ids))
        if any(value < 1 or value > evidence_count for value in citation_ids):
            raise RuntimeError("DeepSeek 返回了不存在的证据编号。")
        if answerable and not citation_ids:
            raise RuntimeError("可回答结果必须至少引用一条证据。")
        if not answerable:
            citation_ids = []

        return GeneratedAnswer(
            answerable=answerable,
            answer=answer.strip(),
            citation_ids=citation_ids,
            reason=reason.strip(),
            model=response.model or self.model,
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            api_calls=api_calls,
        )
