"""对话知识沉淀：经确认的优质问答 → 可检索知识条目，跨会话复用

条目格式（data/knowledge.jsonl，每行一个 JSON）：
  {id, question, answer, citations: [{company, doc_name, page_no}], created_at}
检索单一来源：问题向量相似度 > 阈值即注入上下文。
"""
from __future__ import annotations

import json
import time
import uuid

from .. import config
from ..llm import get_llm

_SIM_THRESHOLD = 0.75   # 问题向量相似度阈值
_vec_cache = None        # [id, vec, entry]


def _load_entries() -> list[dict]:
    if not config.KNOWLEDGE_FILE.exists():
        return []
    out = []
    for line in config.KNOWLEDGE_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def add_knowledge(question: str, answer: str, citations: list[dict] | None = None) -> dict:
    global _vec_cache
    entry = {
        "id": uuid.uuid4().hex[:10],
        "question": question,
        "answer": answer,
        "citations": citations or [],
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(config.KNOWLEDGE_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    _vec_cache = None  # 失效缓存
    return entry


def delete_knowledge(entry_id: str) -> bool:
    global _vec_cache
    entries = _load_entries()
    rest = [e for e in entries if e["id"] != entry_id]
    if len(rest) == len(entries):
        return False
    config.KNOWLEDGE_FILE.write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in rest) + "\n",
        encoding="utf-8",
    )
    _vec_cache = None
    return True


def list_knowledge() -> list[dict]:
    return _load_entries()


def _ensure_vectors(entries) -> list:
    """惰性构建问题向量缓存 [id, vec, entry]"""
    global _vec_cache
    if _vec_cache is not None and len(_vec_cache) == len(entries):
        return _vec_cache
    if not entries:
        _vec_cache = []
        return _vec_cache
    llm = get_llm()
    vecs = llm.embed([e["question"] for e in entries])
    _vec_cache = [(e["id"], vecs[i], e) for i, e in enumerate(entries)]
    return _vec_cache


def search_knowledge(query: str, top_n: int = 2) -> list[dict]:
    """按问题相似度检索沉淀知识；无条目或失败时返回 []"""
    try:
        entries = _load_entries()
        if not entries:
            return []
        items = _ensure_vectors(entries)
        llm = get_llm()
        qvec = llm.embed_query(query)
        scored = sorted(items, key=lambda it: float(it[1] @ qvec), reverse=True)[:top_n]
        out = []
        for _id, vec, e in scored:
            sim = float(vec @ qvec)
            if sim >= _SIM_THRESHOLD:
                out.append({"id": e["id"], "question": e["question"],
                            "answer": e["answer"], "citations": e["citations"], "sim": sim})
        return out
    except Exception:
        return []
