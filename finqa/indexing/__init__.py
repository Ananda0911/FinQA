"""索引：FAISS 向量库 + BM25 词库，按公司独立路由"""
from .vector_store import VectorIndex
from .bm25_store import BM25Index
from .fusion import hybrid_fuse
