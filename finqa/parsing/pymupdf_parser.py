"""PyMuPDF 回退解析器：按页抽取文本 + 表格识别（内建 table finder）

超长年报按 config.PARSE_PAGE_BATCH(200页) 分段逐段解析，避免整本处理超内存。
"""
from __future__ import annotations

import uuid

try:
    import pymupdf as fitz  # PyMuPDF >= 2.0 推荐写法
except ImportError:
    import fitz  # 兼容旧版

from .. import config
from .base import BaseParser, Page, ParsedDocument, TableBlock


class PyMuPDFParser(BaseParser):
    name = "pymupdf"

    def parse(self, pdf_path: str, name: str, doc_id: str | None = None) -> ParsedDocument:
        doc = fitz.open(pdf_path)
        doc_id = doc_id or uuid.uuid4().hex[:12]
        pages: list[Page] = []

        for start in range(0, doc.page_count, config.PARSE_PAGE_BATCH):
            for pno in range(start, min(start + config.PARSE_PAGE_BATCH, doc.page_count)):
                page = doc.load_page(pno)
                text = page.get_text("text").strip()
                tables: list[TableBlock] = []
                try:
                    finder = page.find_tables()
                    for t in finder.tables:
                        rows = [[str(c or "").strip() for c in row] for row in t.extract()]
                        rows = [r for r in rows if any(c for c in r)]
                        if len(rows) >= 2:
                            tables.append(TableBlock(page_no=pno + 1, rows=rows))
                except Exception:
                    pass  # 表格识别失败不影响文本
                pages.append(Page(page_no=pno + 1, text=text, tables=tables))
        doc.close()
        return ParsedDocument(doc_id=doc_id, doc_name=name, pages=pages)
