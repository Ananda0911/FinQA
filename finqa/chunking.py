"""小切片（Small Chunk）构建：页面 → 可检索语义块，保留父页面映射

Small-to-Big 策略：检索单位是 ~500 字符的小切片，喂给 LLM 的上下文则
回捞整页父页面，补全跨页指标与表格上下文。
"""
from __future__ import annotations

import re

from . import config
from .parsing.base import ParsedDocument

# 按 sentence break 优先切分
_SENT_SPLIT = re.compile(r"(?<=[。！？；!?;\n])")

_CITATION_HEADER = "【资料来源：{doc_name} 第{page_no}页】\n"


def _sliding_chunks(text: str, size: int, overlap: int) -> list[str]:
    """按句子边界滑窗切块；块内不足 size 不再切分"""
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    # 先按句切，再组包
    sentences = [s for s in _SENT_SPLIT.split(text) if s and s.strip()]
    chunks: list[str] = []
    buf, buf_len = "", 0
    for sent in sentences:
        if buf_len + len(sent) > size and buf:
            chunks.append(buf.strip())
            # 回退 overlap 字符，保持上下文连续
            buf = buf[-overlap:] if overlap > 0 else ""
            buf_len = len(buf)
        buf += sent
        buf_len = len(buf)
        # 超长单句强制硬切
        while buf_len > size * 1.5:
            chunks.append(buf[:size].strip())
            buf = buf[size - overlap : size] if overlap else ""
            buf_len = len(buf)
    if buf.strip():
        chunks.append(buf.strip())
    return chunks


def chunk_document(
    doc: ParsedDocument,
    chunk_size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
) -> list[dict]:
    """返回切片列表，每个切片保留父页面映射：
    {chunk_id, doc_id, doc_name, page_no, text}
    """
    chunks: list[dict] = []
    for page in doc.pages:
        # 页面头部注：保证嵌入时带上页码与文档名信息
        body = _CITATION_HEADER.format(doc_name=doc.doc_name, page_no=page.page_no) + page.text
        pieces = _sliding_chunks(body, chunk_size, overlap)
        for i, piece in enumerate(pieces):
            chunks.append({
                "chunk_id": f"{doc.doc_id}_p{page.page_no}_c{i}",
                "doc_id": doc.doc_id,
                "doc_name": doc.doc_name,
                "page_no": page.page_no,
                "text": piece,
            })
    return chunks
