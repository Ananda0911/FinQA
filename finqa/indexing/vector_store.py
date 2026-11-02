"""FAISS 向量索引（按公司独立），内积 + L2 归一化 = 余弦相似度"""
from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np


class VectorIndex:
    """meta: [{id, ref}]  id 为 FAISS 内部连续编号，ref 指向切片 chunk_id 或其他负载"""

    def __init__(self, dim: int, meta: list[dict] | None = None):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.meta = meta or []

    def add(self, vectors: np.ndarray, refs: list[dict]):
        """vectors 已 L2 归一化；refs 与 vectors 一一对应"""
        start = len(self.meta)
        self.index.add(np.ascontiguousarray(vectors, dtype=np.float32))
        for i, ref in enumerate(refs):
            self.meta.append({"id": start + i, **ref})

    def search(self, query_vec: np.ndarray, k: int) -> list[dict]:
        if len(self.meta) == 0:
            return []
        k = min(k, len(self.meta))
        scores, ids = self.index.search(query_vec.reshape(1, -1).astype(np.float32), k)
        out = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            out.append({**self.meta[int(idx)], "score": float(score)})
        return out

    # ---------- 持久化 ----------
    # 注意：faiss.write_index 走 C++ fopen，Windows 下不支持含中文的路径，
    # 故用 serialize_index 序列化为字节后由 Python 写文件（原生 Unicode 路径安全）
    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        blob = faiss.serialize_index(self.index)
        path.write_bytes(np.asarray(blob, dtype=np.uint8).tobytes())
        (path.with_suffix(".meta.json")).write_text(
            json.dumps(self.meta, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path, dim: int) -> "VectorIndex":
        vi = cls(dim)
        data = np.frombuffer(path.read_bytes(), dtype=np.uint8)
        vi.index = faiss.deserialize_index(data)
        vi.meta = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
        return vi
