"""解析数据模型与解析器接口"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field


@dataclass
class TableBlock:
    """PDF 中抽取出的表格（rows[0] 为表头）"""
    page_no: int
    caption: str = ""
    rows: list[list[str]] = field(default_factory=list)

    def to_markdown(self) -> str:
        if not self.rows:
            return ""
        lines = []
        for i, row in enumerate(self.rows):
            lines.append("| " + " | ".join(str(c or "").strip() for c in row) + " |")
            if i == 0:
                lines.append("|" + "---|" * len(row))
        return "\n".join(lines)


@dataclass
class Page:
    page_no: int          # 1-based 页码
    text: str
    tables: list[TableBlock] = field(default_factory=list)


@dataclass
class ParsedDocument:
    doc_id: str
    doc_name: str
    pages: list[Page] = field(default_factory=list)

    @property
    def total_pages(self) -> int:
        return len(self.pages)


class BaseParser(abc.ABC):
    """解析器接口：解析超长 PDF，返回按页组织的文本与表格"""

    name = "base"

    @abc.abstractmethod
    def parse(self, pdf_path: str, name: str, doc_id: str) -> ParsedDocument:
        ...
