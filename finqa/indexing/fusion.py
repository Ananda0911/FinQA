"""混合检索融合：BM25（精确分词）与 Vector（语义）加权融合

三路信号归一化后加权：
  - Vector   语义相似
  - BM25     精确关键词/数字
  - Query    Doc2Query 扩展问题命中（若已构建）
"""
from __future__ import annotations

from .. import config


def _minmax(pairs: list[dict]) -> dict:
    """对一路召回的 score 做 min-max 归一化到 [0,1]"""
    if not pairs:
        return {}
    lo = min(p["score"] for p in pairs)
    hi = max(p["score"] for p in pairs)
    rng = (hi - lo) or 1.0
    return {p["chunk_id"]: (p["score"] - lo) / rng for p in pairs}


def hybrid_fuse(
    vector_hits: list[dict],      # [{chunk_id, score, ...}]
    bm25_hits: list[dict],       # [{chunk_id, score, ...}]  已映射到 chunk_id
    query_hits: list[dict] | None = None,  # Doc2Query 命中，score 为问题相似度
) -> list[dict]:
    """返回融合排序后的 [{chunk_id, score, 来源标记}]"""
    nv, nb, nq = _minmax(vector_hits), _minmax(bm25_hits), _minmax(query_hits or [])
    keys: dict[str, dict] = {}

    def bump(cid: str, add: float, src: str):
        e = keys.setdefault(cid, {"chunk_id": cid, "score": 0.0, "sources": []})
        e["score"] += add
        if src not in e["sources"]:
            e["sources"].append(src)

    for cid, s in nv.items():
        bump(cid, s * config.W_VECTOR, "vector")
    for cid, s in nb.items():
        bump(cid, s * config.W_BM25, "bm25")
    for cid, s in nq.items():
        bump(cid, s * config.W_QUERY, "query")
    return sorted(keys.values(), key=lambda e: e["score"], reverse=True)
