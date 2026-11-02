"""BM25 索引：jieba 精确分词，按公司独立"""
from __future__ import annotations

import pickle
from pathlib import Path

import jieba
from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    return [t for t in jieba.lcut(text) if t.strip()]


class BM25Index:
    def __init__(self, corpus: list[str] | None = None):
        self.corpus = corpus or []
        self._bm25: BM25Okapi | None = None
        if self.corpus:
            self._build()

    def _build(self):
        tokenized = [tokenize(t) for t in self.corpus]
        if any(tokenized):
            self._bm25 = BM25Okapi(tokenized)
        else:
            self._bm25 = None

    def add(self, texts: list[str]):
        self.corpus.extend(texts)
        self._build()  # 语料规模为切片级（万级以内），重建可接受

    def search(self, query: str, k: int) -> list[dict]:
        if not self._bm25 or not self.corpus:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        top = sorted(
            ((float(s), i) for i, s in enumerate(scores) if s > 0), reverse=True
        )[:k]
        return [{"id": i, "score": s} for s, i in top]

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps({"corpus": self.corpus}))

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        data = pickle.loads(path.read_bytes())
        idx = cls(data["corpus"])
        return idx
