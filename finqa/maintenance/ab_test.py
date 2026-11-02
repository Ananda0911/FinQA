"""切片版本 A/B 对比：不同 chunk 参数下的检索质量评估

流程：
1. 用 Doc2Query 从现有切片生成评测问题（gold = 原切片）
2. 构建对照索引（大切片 v2）并同样生成评测问题
3. 两版各跑 Top-K 检索，对比 Hit@5 / MRR
产物：data/companies/{company}/ab_report.json
"""
from __future__ import annotations

import json

from .. import config
from ..llm import get_llm
from .doc2query import generate_chunk_questions


def _load_chunks(company: str) -> list[dict]:
    f = config.COMPANIES_DIR / company / "chunks.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else []


def _build_eval_set(chunks: list[dict], llm, n: int) -> list[dict]:
    """从切片生成评测集：问题 + gold chunk_id"""
    import random

    sampled = random.sample(chunks, min(n, len(chunks)))
    evals, done = [], 0
    for c in sampled:
        qs = generate_chunk_questions(c["text"], llm, 2)
        for q in qs:
            evals.append({"question": q, "gold": c["chunk_id"]})
        done += 1
    return evals


def _rebuild_variant(company: str, chunks: list[dict], vdir, llm) -> None:
    """按对照参数重建独立向量索引"""
    from ..indexing import VectorIndex

    vdir.mkdir(parents=True, exist_ok=True)
    vi = VectorIndex(config.EMBED_DIM)
    vecs = llm.embed([c["text"] for c in chunks])
    refs = [{"chunk_id": c["chunk_id"], "pos": i} for i, c in enumerate(chunks)]
    vi.add(vecs, refs)
    vi.save(vdir / "vector.faiss")


def _run_eval(company: str, evals: list[dict], index_dir, k: int = 5) -> dict:
    """在指定索引上跑 Hit@k / MRR（chunk 位置表从对应索引目录加载，v1/v2 各自正确）"""
    from ..indexing import VectorIndex

    vi = VectorIndex.load(index_dir / "vector.faiss", config.EMBED_DIM)
    idata = json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))
    chunk_pos = {c["chunk_id"]: i for i, c in enumerate(idata)}
    llm = get_llm()
    hits, mrr_sum = 0, 0.0
    for e in evals:
        qvec = llm.embed_query(e["question"])
        res = vi.search(qvec, k)
        gold_pos = chunk_pos.get(e["gold"], -1)
        rank = next(
            (r + 1 for r, h in enumerate(res) if h["pos"] == gold_pos), 0
        )
        if rank:
            hits += 1
            mrr_sum += 1.0 / rank
    n = max(len(evals), 1)
    return {"hit@%d" % k: hits / n, "mrr": mrr_sum / n, "n": len(evals)}


def run_ab_test(company: str, eval_n: int = 20, emit=None) -> dict:
    """A/B 主入口：v1(默认切片) vs v2(大切片 800/100)"""
    from ..chunking import _sliding_chunks  # 复用滑窗切块

    llm = get_llm()
    chunks = _load_chunks(company)
    if len(chunks) < 5:
        raise RuntimeError("切片太少，无法做 A/B 评测")
    cdir = config.COMPANIES_DIR / company
    v2dir = cdir / "ab_v2"
    emit and emit("ab", 0, 4)

    # 1. v2 大切片
    pages = {}
    for c in chunks:  # 从切片聚合回页面
        pages.setdefault(c["page_no"], []).append(c)
    v2_chunks = []
    for page_no, cs in pages.items():
        full_text = "\n".join(c["text"] for c in cs)
        pieces = _sliding_chunks(full_text, config.AB_CHUNK_SIZE, config.AB_CHUNK_OVERLAP)
        for i, p in enumerate(pieces):
            v2_chunks.append({
                "chunk_id": f"v2_p{page_no}_c{i}",
                "doc_id": cs[0]["doc_id"], "doc_name": cs[0]["doc_name"],
                "page_no": page_no, "text": p,
                "_v1_ids": sorted(c["chunk_id"] for c in cs),  # list：set 不可 JSON 序列化
            })
    emit and emit("ab", 1, 4)
    _rebuild_variant(company, v2_chunks, v2dir, llm)

    # 2. 评测集（各版独立生成，gold 映射到可对比粒度）
    emit and emit("ab", 2, 4)
    evals_v1 = _build_eval_set(chunks, llm, eval_n)
    # v2 的 gold：把 v1 gold 切片映射到覆盖它的 v2 切片
    evals_v2 = []
    for e in evals_v1:
        cover = [v["chunk_id"] for v in v2_chunks
                 if e["gold"] in v.get("_v1_ids", {})]
        if cover:
            evals_v2.append({"question": e["question"], "gold": cover[0]})
    (v2dir / "chunks.json").write_text(json.dumps(v2_chunks, ensure_ascii=False), encoding="utf-8")

    # 3. 双版检索评测
    emit and emit("ab", 3, 4)
    r1 = _run_eval(company, evals_v1, cdir, k=5)
    emit and emit("ab", 4, 4)
    v2run = _run_eval(company, evals_v2, v2dir, k=5)

    report = {
        "company": company,
        "v1": {"chunk_size": config.CHUNK_SIZE, "overlap": config.CHUNK_OVERLAP, **r1},
        "v2": {"chunk_size": config.AB_CHUNK_SIZE, "overlap": config.AB_CHUNK_OVERLAP, **v2run},
        "verdict": ("v2 更优" if v2run["mrr"] > r1["mrr"]
                    else "v1 更优" if r1["mrr"] > v2run["mrr"] else "持平"),
    }
    (cdir / "ab_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return report
