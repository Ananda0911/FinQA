"""GraphRAG Local / Global Search

- Local：问题命中图节点 → 一跳邻域 + 关联切片，作为局部上下文增强
- Global：社区摘要与问题做向量相关性排序 → 取 Top-N 摘要作为全局背景
两者均以「额外上下文」形式并入统一的分题型生成链路。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .. import config
from ..llm import get_llm


def _load_graph(company: str) -> dict | None:
    f = config.GRAPH_DIR / company / "graph.json"
    if not f.exists():
        return None
    return json.loads(f.read_text(encoding="utf-8"))


# ---------------- Global Search ----------------
def global_context(query: str, companies: list[str]) -> str:
    """全局检索：取相关性最高的社区摘要，拼成全局背景文本"""
    llm = get_llm()
    summaries: list[str] = []
    for company in companies:
        g = _load_graph(company)
        if not g:
            continue
        comms = [c for c in g.get("communities", []) if c.get("summary")]
        if not comms:
            continue
        if len(comms) <= config.GLOBAL_COMMUNITY_TOP:
            summaries.extend(c["summary"] for c in comms)
            continue
        qvec = llm.embed_query(query)
        svecs = llm.embed([c["summary"] for c in comms])
        sims = svecs @ qvec
        top = np.argsort(-sims)[: config.GLOBAL_COMMUNITY_TOP]
        summaries.extend(comms[i]["summary"] for i in top)
    return "\n\n".join(
        f"◆ 社区{i + 1}摘要：{s}" for i, s in enumerate(summaries)
    )


# ---------------- Local Search ----------------
def local_context(query: str, company: str, top_nodes: int = 8) -> str:
    """局部检索：抽取问题实体 → 命中图节点的一跳邻域 + 关联切片摘要"""
    llm = get_llm()
    g = _load_graph(company)
    if not g:
        return ""
    nodes, edges = g["nodes"], g["edges"]
    prompt = f"""从下面的问题中抽取可能在财报知识图谱中出现的实体名（财务指标、业务、子公司、风险等），输出 JSON 数组。
问题：{query}
输出格式：{{"entities": ["实体名", ...]}}"""
    try:
        data = llm.chat_json(prompt)
        names = [str(n).strip() for n in data.get("entities", []) if str(n).strip()]
    except Exception:
        names = []

    matched: list[str] = []
    for n in names:
        for node_name in nodes:
            if n and (n in node_name or node_name in n) and node_name not in matched:
                matched.append(node_name)
    if not matched:
        return ""

    # 一跳邻域 + 关联切片 id
    chunk_ids: set[str] = set()
    rel_lines: list[str] = []
    for e in edges:
        if e["source"] in matched or e["target"] in matched:
            rel_lines.append(f"{e['source']} — {e['target']}：{e['description']}")
            if e.get("chunk_id"):
                chunk_ids.add(e["chunk_id"])
    for m in matched[:top_nodes]:
        chunk_ids.update(nodes[m].get("chunks", [])[:3])

    node_lines = [
        f"- {m}（{nodes[m].get('type', '')}）：{nodes[m].get('description', '')}"
        for m in matched[:top_nodes]
    ]
    return "\n".join(
        [f"【{company} 相关实体】"] + node_lines
        + ["【相关关系】"] + (rel_lines[:20] or ["（无）"])
    )
