"""MinerU 解析器（中文财报首选，替代英文导向的 Docling）

通过 MinerU CLI 解析 PDF：
  mineru -p <pdf> -o <outdir> -b pipeline
输出目录下找 <name>_content.json（含每页布局块），文本块取 text，
表格块取 table_body（HTML）并转为行列结构。content.json 缺失时回退读取
markdown 输出。超长年报按 200 页分段拆分 PDF 后逐段解析。
"""
from __future__ import annotations

import html
import json
import re
import shutil
import subprocess
import tempfile
import uuid
from html.parser import HTMLParser
from pathlib import Path

try:
    import pymupdf as fitz  # PyMuPDF >= 2.0 推荐写法
except ImportError:
    import fitz  # 兼容旧版

from .. import config
from .base import BaseParser, Page, ParsedDocument, TableBlock


class _HTMLTableParser(HTMLParser):
    """极简 HTML 表格转行列（处理 rowspan/colspan 展开）"""

    def __init__(self):
        super().__init__()
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None
        self._spans: dict[tuple[int, int], str] = {}  # 占位展开
        self._r = 0

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._row = []
            self._r = len(self.rows)
        elif tag == "td" or tag == "th":
            self._cell = []

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._cell is not None:
            text = " ".join("".join(self._cell).split())
            self._row and self._row.append(text)
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None


def html_table_to_rows(table_html: str) -> list[list[str]]:
    p = _HTMLTableParser()
    p.feed(table_html)
    rows = [r for r in p.rows if any(c for c in r)]
    return rows


def _minify_html(html_text: str) -> str:
    html_text = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html_text, flags=re.S)
    return re.sub(r"\s+", " ", html_text)


class MinerUParser(BaseParser):
    name = "mineru"

    def available(self) -> bool:
        return shutil.which("mineru") is not None or shutil.which("magic-pdf") is not None

    # ---------- CLI ----------
    def _run_mineru(self, pdf: Path, outdir: Path) -> Path | None:
        for cli in ("mineru", "magic-pdf"):
            if shutil.which(cli):
                cmd = [cli]
                if cli == "mineru":
                    cmd += ["-p", str(pdf), "-o", str(outdir), "-b", "pipeline"]
                else:  # 旧版 magic-pdf CLI
                    cmd += ["-p", str(pdf), "-o", str(outdir)]
                try:
                    subprocess.run(
                        cmd, check=True, capture_output=True, timeout=1800
                    )  # noqa
                    return outdir
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                    continue
        return None

    # ---------- 输出解析 ----------
    def _parse_content_json(self, json_path: Path, total: int) -> list[Page] | None:
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if not isinstance(data, list):
            return None
        pages: list[Page] = []
        for i, page in enumerate(data):
            texts, tables = [], []
            for block in page.get("items", page.get("blocks", [])):
                btype = block.get("type", "")
                if btype in ("text", "title", "image_caption", "table_caption", "interline_equation"):
                    t = block.get("text") or ""
                    if t.strip():
                        texts.append(t.strip())
                elif btype == "table":
                    body = block.get("table_body") or ""
                    if isinstance(body, dict):
                        body = body.get("html", "")
                    rows = html_table_to_rows(_minify_html(body)) if body else []
                    if len(rows) >= 2:
                        caption = (block.get("table_caption") or [""])[0] \
                            if isinstance(block.get("table_caption"), list) else str(block.get("table_caption") or "")
                        tables.append(TableBlock(page_no=i + 1, caption=caption.strip(), rows=rows))
            pages.append(Page(page_no=i + 1, text="\n".join(texts), tables=tables))
        while len(pages) < total:  # content.json 可能缺页
            pages.append(Page(page_no=len(pages) + 1, text=""))
        return pages

    def _parse_markdown(self, md_path: Path, total: int) -> list[Page]:
        """无 content.json 时读 markdown，整本暂记为无页码文本（页码置 1）"""
        text = md_path.read_text(encoding="utf-8")
        return [Page(page_no=1, text=text)]

    # ---------- 主流程 ----------
    def parse(self, pdf_path: str, name: str, doc_id: str | None = None) -> ParsedDocument:
        if not self.available():
            raise RuntimeError("MinerU 不可用，请 pip install 'mineru[core]' 或改用 PyMuPDF 后端")
        doc_id = doc_id or uuid.uuid4().hex[:12]
        pdf = Path(pdf_path)
        src = fitz.open(pdf)
        total = src.page_count
        all_pages: list[Page] = [Page(page_no=i + 1, text="") for i in range(total)]

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            for start in range(0, total, config.PARSE_PAGE_BATCH):
                end = min(start + config.PARSE_PAGE_BATCH, total)
                part = tmpdir / f"_part_{start}.pdf"
                outdir = tmpdir / f"out_{start}"
                part_doc = fitz.open()
                part_doc.insert_pdf(src, from_page=start, to_page=end - 1)
                part_doc.save(part)
                part_doc.close()
                if self._run_mineru(part, outdir) is None:
                    continue
                pages = self._extract_segment(outdir, end - start)
                if pages:
                    for j, p in enumerate(pages):
                        if p.text or p.tables:
                            all_pages[start + j] = p
        src.close()
        return ParsedDocument(doc_id=doc_id, doc_name=name, pages=all_pages)

    def _extract_segment(self, outdir: Path, expected: int) -> list[Page] | None:
        jsons = list(outdir.rglob("*_content.json")) + list(outdir.rglob("*content*.json"))
        for jp in jsons:
            pages = self._parse_content_json(jp, expected)
            if pages:
                return pages
        mds = list(outdir.rglob("*.md"))
        if mds:
            return self._parse_markdown(mds[0], expected)
        return None
