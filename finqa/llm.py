"""DashScope 统一封装：Chat / Embedding / Rerank / JSON 输出

- Chat 与 Embedding 走 OpenAI 兼容模式
- Rerank 走 dashscope 原生 TextReRank 接口
"""
from __future__ import annotations

import json
import re
from typing import Iterable

import numpy as np

from . import config


class LLMError(RuntimeError):
    pass


def _require_key():
    if not config.DASHSCOPE_API_KEY:
        raise LLMError(
            "未配置 DASHSCOPE_API_KEY。请复制 .env.example 为 .env 并填入阿里云百炼 API Key。"
        )


class LLMClient:
    def __init__(self):
        _require_key()
        from openai import OpenAI

        self._client = OpenAI(
            api_key=config.DASHSCOPE_API_KEY,
            base_url=config.DASHSCOPE_BASE_URL,
        )

    # ---------- Chat ----------
    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.2,
        max_tokens: int = 2048,
        model: str | None = None,
    ) -> str:
        resp = self._client.chat.completions.create(
            model=model or config.LLM_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    def chat_stream(
        self,
        messages: list[dict],
        temperature: float = 0.2,
        max_tokens: int = 2048,
        model: str | None = None,
    ):
        """流式生成：逐段 yield 增量文本（用于前端打字机效果，大幅降低体感延迟）"""
        stream = self._client.chat.completions.create(
            model=model or config.LLM_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in stream:
            choices = chunk.choices or []
            if choices and choices[0].delta and choices[0].delta.content:
                yield choices[0].delta.content

    def chat_json(self, prompt: str, system: str | None = None, temperature: float = 0.1,
                  model: str | None = None) -> dict:
        """要求模型输出 JSON 并解析。失败时抛出 LLMError。"""
        msgs: list[dict] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt + "\n\n只输出一个合法 JSON 对象，不要输出任何其他文字或代码块标记。"})
        text = self.chat(msgs, temperature=temperature, model=model)
        return parse_json(text)

    # ---------- Embedding ----------
    def embed(self, texts: Iterable[str]) -> np.ndarray:
        texts = [t if t.strip() else " " for t in texts]
        vecs: list[list[float]] = []
        for i in range(0, len(texts), config.EMBED_BATCH):
            batch = texts[i : i + config.EMBED_BATCH]
            resp = self._client.embeddings.create(model=config.EMBED_MODEL, input=batch)
            data = sorted(resp.data, key=lambda d: d.index)
            vecs.extend(d.embedding for d in data)
        arr = np.asarray(vecs, dtype=np.float32)
        # L2 归一化，配合内积索引做余弦相似度
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    # ---------- Rerank ----------
    def rerank(self, query: str, documents: list[str], top_n: int | None = None) -> list[dict]:
        """返回 [{'index': i, 'relevance_score': s}, ...] 按分数降序"""
        if not documents:
            return []
        import dashscope

        resp = dashscope.TextReRank.call(
            model=config.RERANK_MODEL,
            query=query,
            documents=documents,
            top_n=top_n or len(documents),
            api_key=config.DASHSCOPE_API_KEY,
            return_documents=False,
        )
        if resp.status_code != 200:
            raise LLMError(f"Rerank 调用失败: {resp.code} {resp.message}")
        return [
            {"index": r["index"], "relevance_score": r["relevance_score"]}
            for r in resp.output["results"]
        ]


def parse_json(text: str) -> dict:
    """容错解析模型输出的 JSON（剥掉代码块围栏、截取首尾大括号）"""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise LLMError(f"模型未输出 JSON: {text[:200]}")
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError as e:
        raise LLMError(f"JSON 解析失败: {e}\n{text[:300]}") from e


# 模块级单例（惰性）
_client: LLMClient | None = None


def get_llm() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client
