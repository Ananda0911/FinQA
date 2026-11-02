"""LLM 实体与关系抽取（并发批量）"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from .. import config
from ..llm import LLMClient, LLMError, parse_json

_EXTRACT_PROMPT = """你是财务知识图谱构建助手。请从下面的财报片段中抽取实体与关系。

抽取要求：
1. 实体类型限：公司/子公司、产品或业务、财务指标、人、风险因素、地区/市场；
2. 实体名用原文规范表述（如"营业收入"、"应收账款风险"）；
3. 关系只抽取文中明确表达的（如"XX公司的主要产品是YY"、"YY业务收入占比30%"）；
4. 没有实体时输出空数组。

财报片段：
{chunk}

输出 JSON：
{{"entities": [{{"name": "...", "type": "...", "description": "一句话描述"}}],
"relations": [{{"source": "实体名", "target": "实体名", "description": "关系描述"}}]}}"""


def extract_from_chunk(chunk: dict, llm: LLMClient) -> dict:
    """从单个切片抽取 {entities: [...], relations: [...], chunk_id}"""
    try:
        data = llm.chat_json(_EXTRACT_PROMPT.format(chunk=chunk["text"][:3000]))
    except (LLMError, Exception):
        return {"entities": [], "relations": [], "chunk_id": chunk.get("chunk_id", "")}

    def _norm(items):
        out = []
        for e in items or []:
            if isinstance(e, dict) and e.get("name"):
                out.append(e)
        return out

    return {
        "entities": _norm(data.get("entities")),
        "relations": _norm(data.get("relations")),
        "chunk_id": chunk.get("chunk_id", ""),
    }


def extract_from_chunks(chunks: list[dict], llm: LLMClient,
                        emit=None) -> tuple[dict, list[dict]]:
    """并发抽取全部切片，返回 (节点表, 边表)

    节点表: {name: {"type", "description", "chunks": set(chunk_id)}}
    边表:   [{"source", "target", "description", "chunk_id"}]
    """
    nodes: dict = {}
    edges: list = []
    total = len(chunks)
    done = 0
    with ThreadPoolExecutor(max_workers=config.GRAPH_WORKERS) as pool:
        futures = {pool.submit(extract_from_chunk, c, llm): c for c in chunks}
        for fut in as_completed(futures):
            done += 1
            emit and emit("extract", done, total)
            r = fut.result()
            for e in r["entities"]:
                name = e["name"].strip()
                if not name:
                    continue
                node = nodes.setdefault(
                    name, {"type": e.get("type", ""), "description": e.get("description", ""), "chunks": []}
                )
                if r["chunk_id"] and r["chunk_id"] not in node["chunks"]:
                    node["chunks"].append(r["chunk_id"])
            for rel in r["relations"]:
                s, t = str(rel.get("source", "")).strip(), str(rel.get("target", "")).strip()
                if s and t and s in nodes and t in nodes:
                    edges.append({
                        "source": s, "target": t,
                        "description": rel.get("description", ""), "chunk_id": r["chunk_id"],
                    })
    return nodes, edges
