"""Doc2Query：为切片生成扩展问题，提升召回覆盖"""
from __future__ import annotations

from ..llm import LLMClient

_PROMPT = """下面是上市公司财报知识库的一个切片。请站在「用户会怎么问」的角度，生成{n}个多样化的问题：
- 问题必须能由该切片内容直接回答；
- 覆盖不同问法（数值、是否、列举、口语化改写）；
- 不要出现"资料/切片/上文"等字样。

切片内容：
{chunk}

输出 JSON：{{"questions": ["问题1", "问题2", ...]}}"""


def generate_chunk_questions(chunk_text: str, llm: LLMClient, n: int = 3) -> list[str]:
    try:
        data = llm.chat_json(_PROMPT.format(n=n, chunk=chunk_text[:3000]))
        qs = [q.strip() for q in data.get("questions", []) if isinstance(q, str) and q.strip()]
        return qs[:n]
    except Exception:
        return []
