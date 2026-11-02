"""图谱构建：实体图 → 社区聚类（模块度）→ 社区摘要，落盘持久化"""
from __future__ import annotations

import json
from pathlib import Path

import networkx as nx

from .. import config
from ..llm import LLMClient
from .extract import extract_from_chunks


def _graph_dir(company: str) -> Path:
    d = config.GRAPH_DIR / company
    d.mkdir(parents=True, exist_ok=True)
    return d


def graph_exists(company: str) -> bool:
    return (_graph_dir(company) / "graph.json").exists()


def build_graph(company: str, llm: LLMClient, emit=None) -> dict:
    """建图主流程：抽样切片 → 实体抽取 → 社区聚类 → 社区摘要

    返回统计信息 dict；产物保存在 data/graph/{company}/graph.json
    graph.json: {nodes: {...}, edges: [...], communities: [{id, nodes, summary}]}
    """
    import random

    cdir = config.COMPANIES_DIR / company
    chunks = json.loads((cdir / "chunks.json").read_text(encoding="utf-8")) \
        if (cdir / "chunks.json").exists() else []
    if not chunks:
        raise RuntimeError(f"公司 {company} 无切片，请先入库文档")
    if len(chunks) > config.GRAPH_MAX_CHUNKS:
        chunks = random.sample(chunks, config.GRAPH_MAX_CHUNKS)  # 均匀抽样控成本

    nodes, edges = extract_from_chunks(chunks, llm, emit)

    # 无向图做社区聚类
    G = nx.Graph()
    for name, meta in nodes.items():
        G.add_node(name, **meta)
    for e in edges:
        if G.has_edge(e["source"], e["target"]):
            G[e["source"]][e["target"]]["weight"] += 1
        else:
            G.add_edge(e["source"], e["target"], weight=1)

    communities: list[dict] = []
    if G.number_of_nodes() > 0:
        try:
            comms = nx.community.greedy_modularity_communities(G, weight="weight")
        except Exception:
            comms = [set(G.nodes())]
        for i, comm in enumerate(comms):
            members = sorted(comm)
            summary = summarize_community(company, members, nodes, edges, llm)
            communities.append({"id": i, "nodes": members, "summary": summary})

    data = {
        "nodes": {k: {"type": v["type"], "description": v["description"],
                      "chunks": v["chunks"]} for k, v in nodes.items()},
        "edges": edges,
        "communities": communities,
    }
    gdir = _graph_dir(company)
    (gdir / "graph.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return {
        "company": company,
        "nodes": len(nodes), "edges": len(edges),
        "communities": len(communities), "sampled_chunks": len(chunks),
    }


def summarize_community(company: str, members: list[str], nodes: dict,
                        edges: list, llm: LLMClient) -> str:
    """LLM 生成社区摘要（该实体簇讲了什么）"""
    desc = "\n".join(
        f"- {m}（{nodes.get(m, {}).get('type', '')}）：{nodes.get(m, {}).get('description', '')}"
        for m in members[:40]
    )
    rels = [
        f"- {e['source']} — {e['target']}：{e['description']}"
        for e in edges if e["source"] in members and e["target"] in members
    ][:40]
    prompt = f"""请用3-6句中文概括以下「{company}」财报实体社区的整体主题与要点，供后续全局归纳问答使用。

实体：
{desc}

内部关系：
{chr(10).join(rels) or '（无）'}

直接输出概括文字，不要列表编号。"""
    try:
        return llm.chat([{"role": "user", "content": prompt}], temperature=0.2, max_tokens=500)
    except Exception:
        return "；".join(members[:10])
