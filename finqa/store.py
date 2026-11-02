"""KnowledgeBase：核心知识库门面

目录结构 data/companies/{company}/
  chunks.json      所有小切片（含父页面映射）
  pages.json       父页面全文（Small-to-Big 回捞）
  vector.faiss     向量索引
  bm25.pkl         BM25 索引
  qvector.faiss    Doc2Query 问题扩展索引（可选）
  docs.json        已入库文档清单

公司即路由：不同公司的索引完全隔离，检索时按公司目录加载。
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Callable

from . import config
from .chunking import chunk_document
from .indexing import BM25Index, VectorIndex, hybrid_fuse
from .llm import get_llm
from .parsing import get_parser
from .parsing.base import ParsedDocument
from .parsing.table_serialize import serialize_tables


def _company_dir(company: str) -> Path:
    d = config.COMPANIES_DIR / company
    d.mkdir(parents=True, exist_ok=True)
    return d


def _read_json(path: Path, default):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default
    return default


def _write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


class RetrievedChunk(dict):
    """携带父页面上下文的检索单元"""
    # keys: chunk_id, doc_id, doc_name, page_no, company,
    #       chunk_text, page_text, score, sources


class KnowledgeBase:
    # ---------------- 文档入库 ----------------
    def ingest_pdf(
        self,
        company: str,
        pdf_path: str | Path,
        backend: str = "auto",
        do_table_serialize: bool = True,
        do_doc2query: bool = False,
        doc_name: str | None = None,
        progress: Callable[[str, int, int], None] | None = None,
    ) -> dict:
        """入库一份 PDF：解析 → 表格序列化 → 分块 → 三路索引

        doc_name: 显示用文档名（用于引用溯源），缺省取文件名；
        同名文档重新入库时自动替换旧数据（避免重复索引）。
        """
        pdf_path = Path(pdf_path)
        name = doc_name or pdf_path.stem
        doc_id = uuid.uuid4().hex[:12]
        emit = lambda stage, i, n: progress and progress(stage, i, n)

        # 1. 解析
        emit("parse", 0, 1)
        parser = get_parser(backend)
        doc: ParsedDocument = parser.parse(str(pdf_path), name, doc_id)

        # 2. LLM 表格序列化
        if do_table_serialize:
            n_tables = sum(len(p.tables) for p in doc.pages)
            if n_tables:
                def _prog(done, total):
                    emit("table", done, total)
                serialize_tables(doc, progress=_prog)

        # 3. 分块（小切片）
        emit("chunk", 0, 1)
        chunks = chunk_document(doc)
        if not chunks:
            raise RuntimeError(f"未能从 {name} 中解析出任何文本，请检查 PDF")

        # 4. 父页面存储（同名文档替换旧数据）
        cdir = _company_dir(company)
        pages = _read_json(cdir / "pages.json", {})
        docs = _read_json(cdir / "docs.json", [])
        old_ids = {d["doc_id"] for d in docs if d["name"] == name}
        for oid in old_ids:
            pages.pop(oid, None)
        docs = [d for d in docs if d["name"] != name]
        pages[doc_id] = {
            p.page_no: {"doc_name": name, "text": p.text} for p in doc.pages if p.text
        }
        _write_json(cdir / "pages.json", pages)

        # 5. 合并切片（剔除被替换旧文档的切片）
        all_chunks = _read_json(cdir / "chunks.json", [])
        all_chunks = [c for c in all_chunks if c["doc_id"] not in old_ids]
        existing_ids = {c["chunk_id"] for c in all_chunks}
        all_chunks.extend(c for c in chunks if c["chunk_id"] not in existing_ids)
        _write_json(cdir / "chunks.json", all_chunks)

        # 6. 重建三路索引
        emit("index", 0, 3)
        self._rebuild_indexes(company, chunks, emit)
        if do_doc2query:
            self.build_doc2query(company, emit=emit)

        docs.append({"doc_id": doc_id, "name": name, "pages": doc.total_pages,
                     "chunks": len(chunks)})
        _write_json(cdir / "docs.json", docs)
        emit("done", 1, 1)
        return {"doc_id": doc_id, "name": name, "pages": doc.total_pages, "chunks": len(chunks)}

    def _rebuild_indexes(self, company: str, new_chunks: list[dict],
                         emit=None, chunk_store_path: Path | None = None):
        """全量重建向量/BM25 索引（切片规模为万级以内，全量重建成本可接受），
        meta 的 pos 与 chunks.json 顺序严格对齐"""
        cdir = _company_dir(company)
        store_path = chunk_store_path or (cdir / "chunks.json")
        all_chunks = _read_json(store_path, [])

        llm = get_llm()
        texts = [c["text"] for c in all_chunks]
        # 向量索引
        emit and emit("index", 1, 3)
        refs = [{"chunk_id": c["chunk_id"], "pos": i} for i, c in enumerate(all_chunks)]
        vecs = llm.embed(texts)
        vi = VectorIndex(config.EMBED_DIM)
        vi.add(vecs, refs)
        vi.save(cdir / "vector.faiss")

        # BM25 索引
        emit and emit("index", 2, 3)
        bm25 = BM25Index(texts)
        bm25.save(cdir / "bm25.pkl")
        return all_chunks

    # ---------------- Doc2Query 问题扩展 ----------------
    def build_doc2query(self, company: str, max_questions_per_chunk: int = 3,
                        emit=None) -> int:
        """为每个切片生成扩展问题并建索引，提升召回"""
        from .maintenance.doc2query import generate_chunk_questions
        cdir = _company_dir(company)
        chunks = _read_json(cdir / "chunks.json", [])
        qa = _read_json(cdir / "doc2query.json", {})
        todo = [c for c in chunks if c["chunk_id"] not in qa]
        llm = get_llm()
        total = len(todo)
        for i, c in enumerate(todo):
            emit and emit("doc2query", i, total)
            qs = generate_chunk_questions(c["text"], llm, max_questions_per_chunk)
            qa[c["chunk_id"]] = qs
            if (i + 1) % 20 == 0:
                _write_json(cdir / "doc2query.json", qa)
        _write_json(cdir / "doc2query.json", qa)

        # 问题向量索引: ref 指回 chunk_id
        emit and emit("doc2query", total, total)
        qtexts, refs = [], []
        for cid, qs in qa.items():
            for j, q in enumerate(qs):
                qtexts.append(q)
                refs.append({"chunk_id": cid, "q": q})
        if not qtexts:
            return 0
        vecs = llm.embed(qtexts)
        qvi = VectorIndex(config.EMBED_DIM)
        qvi.add(vecs, refs)
        qvi.save(cdir / "qvector.faiss")
        return len(qtexts)

    # ---------------- 公司管理 ----------------
    def list_companies(self) -> list[str]:
        if not config.COMPANIES_DIR.exists():
            return []
        return sorted(
            d.name for d in config.COMPANIES_DIR.iterdir()
            if d.is_dir() and (d / "chunks.json").exists()
        )

    def company_stats(self, company: str) -> dict:
        cdir = _company_dir(company)
        chunks = _read_json(cdir / "chunks.json", [])
        docs = _read_json(cdir / "docs.json", [])
        qn = len(_read_json(cdir / "doc2query.json", {}))
        return {"company": company, "docs": docs, "chunks": len(chunks), "doc2query_chunks": qn}

    # ---------------- 混合检索 ----------------
    def retrieve(
        self,
        query: str,
        companies: str | list[str],
        top_k: int = config.RECALL_TOP_K,
        rerank_top_k: int = config.FINAL_TOP_K,
        do_rerank: bool = True,
        parent_page_cap: int = config.PARENT_PAGE_CAP,
    ) -> list[RetrievedChunk]:
        """完整检索链路：
        公司路由 → 三路召回（Vector/BM25/Doc2Query）→ 加权融合
        → Rerank 精排 → Small-to-Big 父页面回捞
        """
        if isinstance(companies, str):
            companies = [companies]
        llm = get_llm()
        candidates: dict[str, dict] = {}

        for company in companies:
            cdir = config.COMPANIES_DIR / company
            chunks = _read_json(cdir / "chunks.json", [])
            if not chunks:
                continue
            pos_by_chunk = {c["chunk_id"]: i for i, c in enumerate(chunks)}

            # Vector
            vhits = []
            vpath = cdir / "vector.faiss"
            if vpath.exists():
                vi = VectorIndex.load(vpath, config.EMBED_DIM)
                qvec = llm.embed_query(query)
                vhits = [{"chunk_id": h["chunk_id"], "score": h["score"]}
                         for h in vi.search(qvec, top_k) if h["chunk_id"] in pos_by_chunk]
            # BM25
            bhits = []
            bpath = cdir / "bm25.pkl"
            if bpath.exists():
                bm25 = BM25Index.load(bpath)
                bhits = [{"chunk_id": chunks[h["id"]]["chunk_id"], "score": h["score"]}
                         for h in bm25.search(query, top_k)]
            # Doc2Query
            qhits = []
            qpath = cdir / "qvector.faiss"
            if qpath.exists():
                qvi = VectorIndex.load(qpath, config.EMBED_DIM)
                qvec = llm.embed_query(query)
                for h in qvi.search(qvec, 10):
                    cid = h["chunk_id"]
                    if cid in pos_by_chunk:
                        qhits.append({"chunk_id": cid, "score": h["score"]})

            for e in hybrid_fuse(vhits, bhits, qhits):
                cid = e["chunk_id"]
                c = chunks[pos_by_chunk[cid]]
                key = f"{company}::{cid}"
                cur = candidates.get(key)
                if cur is None or e["score"] > cur["score"]:
                    candidates[key] = {
                        "chunk_id": cid,
                        "company": company,
                        "doc_id": c["doc_id"],
                        "doc_name": c["doc_name"],
                        "page_no": c["page_no"],
                        "chunk_text": c["text"],
                        "score": e["score"],
                        "sources": e["sources"],
                    }
        if not candidates:
            return []

        ranked = sorted(candidates.values(), key=lambda e: e["score"], reverse=True)[:top_k]

        # Rerank 精排（粗召回压缩到 Top-3/5）
        if do_rerank and len(ranked) > 1:
            try:
                rr = llm.rerank(query, [r["chunk_text"] for r in ranked], top_n=len(ranked))
                for r in rr:
                    ranked[r["index"]]["rr_score"] = r["relevance_score"]
                ranked.sort(key=lambda e: e.get("rr_score", e["score"]), reverse=True)
            except Exception:
                pass  # rerank 失败则使用融合分
        ranked = ranked[:rerank_top_k]

        # Small-to-Big：回捞父页面，合并同页切片
        pages_cache = {}
        merged: dict[str, RetrievedChunk] = {}
        for r in ranked:
            pkey = f"{r['company']}::{r['doc_id']}::{r['page_no']}"
            if (r["company"], r["doc_id"]) not in pages_cache:
                pages_cache[(r["company"], r["doc_id"])] = _read_json(
                    config.COMPANIES_DIR / r["company"] / "pages.json", {}
                ).get(r["doc_id"], {})
            ptext = pages_cache[(r["company"], r["doc_id"])].get(str(r["page_no"]), {}).get("text", "")
            if pkey in merged:
                continue
            merged[pkey] = RetrievedChunk(
                page_text=ptext or r["chunk_text"],
                **{k: v for k, v in r.items() if k != "chunk_text"},
            )
        return list(merged.values())[:parent_page_cap]


# 模块级单例
_kb: KnowledgeBase | None = None


def get_kb() -> KnowledgeBase:
    global _kb
    if _kb is None:
        _kb = KnowledgeBase()
    return _kb
