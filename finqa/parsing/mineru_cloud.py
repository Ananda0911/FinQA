"""云端 MinerU 解析器：mineru.net 精准解析 API

流程：申请上传链接 → PUT 上传 PDF → 轮询解析结果 → 下载 zip →
     解析 content_list.json 构建按页结构（文本 + 表格）。

特性：
- 表格/公式/多栏版面识别质量远好于 PyMuPDF，适合复杂年报
- 超过 600 页的文档自动分片上传（官方单文件页数上限 600）
- 结果按文件内容哈希落盘缓存，同一 PDF 重新入库不重复消耗云端额度
- 官方文档: https://mineru.net/doc/docs/
"""
from __future__ import annotations

import hashlib
import html as html_mod
import io
import json
import re
import time
import zipfile
from pathlib import Path

import requests

from .. import config
from .base import BaseParser, Page, ParsedDocument, TableBlock

_BASE = "https://mineru.net/api/v4"
# 官方单文件页数实际上限 200 页（vlm 模型），超限自动分片
_CHUNK_PAGES = 200
_POLL_INTERVAL = 6      # 轮询间隔（秒）
_POLL_TIMEOUT = 2400    # 最长等待 40 分钟


def _require_token() -> str:
    if not config.MINERU_API_TOKEN:
        raise RuntimeError(
            "未配置 MINERU_API_TOKEN。请在 https://mineru.net 注册后，"
            "到「API 令牌管理」申请 Token，并填入 .env 的 MINERU_API_TOKEN。"
        )
    return config.MINERU_API_TOKEN


def _headers() -> dict:
    return {"Content-Type": "application/json",
            "Authorization": f"Bearer {config.MINERU_API_TOKEN}"}


def _html_table_rows(table_html: str) -> list[list[str]]:
    """把 MinerU 输出的 HTML 表格还原为行列结构"""
    rows = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", table_html, re.S | re.I):
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, re.S | re.I)
        cells = [html_mod.unescape(re.sub(r"<[^>]+>", "", c)).replace("\xa0", " ").strip()
                 for c in cells]
        if any(cells):
            rows.append(cells)
    return rows


def _content_list_to_pages(items: list[dict], page_offset: int) -> list[Page]:
    """content_list 条目按 page_idx 聚合成页面；page_idx 为分片内 0 基页码"""
    pages: dict[int, Page] = {}
    for it in items:
        pno = int(it.get("page_idx", 0)) + page_offset
        page = pages.setdefault(pno, Page(page_no=pno, text="", tables=[]))
        t = it.get("type", "text")
        if t == "text":
            txt = (it.get("text") or "").strip()
            if txt:
                page.text = (page.text + "\n" + txt).strip()
        elif t == "table":
            caption_parts = it.get("table_caption") or []
            if isinstance(caption_parts, str):
                caption_parts = [caption_parts]
            rows = _html_table_rows(it.get("table_body") or "")
            rows = [r for r in rows if any(c for c in r)]
            if len(rows) >= 2:
                page.tables.append(TableBlock(
                    page_no=pno + 1, caption=" ".join(caption_parts).strip(), rows=rows))
        elif t == "image":
            cap = it.get("img_caption") or []
            if isinstance(cap, str):
                cap = [cap]
            txt = " ".join(cap).strip()
            if txt:
                page.text = (page.text + "\n" + txt).strip()
        # equation 等其他类型：忽略（财报场景无实质影响）
    return sorted(pages.values(), key=lambda p: p.page_no)


class CloudMinerUParser(BaseParser):
    name = "mineru_cloud"

    def parse(self, pdf_path: str, name: str, doc_id: str | None = None) -> ParsedDocument:
        import uuid
        doc_id = doc_id or uuid.uuid4().hex[:12]
        _require_token()
        pdf_path = Path(pdf_path)
        data = pdf_path.read_bytes()

        # 1. 解析结果缓存（按内容哈希，重新入库不重复扣页数）
        cache_dir = config.CACHE_DIR / "mineru_parse"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / (hashlib.md5(data).hexdigest() + ".json")
        if cache_file.exists():
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            pages = [Page(page_no=p["page_no"], text=p["text"],
                          tables=[TableBlock(page_no=p["page_no"], caption=t.get("caption", ""),
                                             rows=t.get("rows", []))
                                  for t in p.get("tables", [])])
                     for p in cached["pages"]]
            return ParsedDocument(doc_id=doc_id, doc_name=name, pages=pages)

        # 2. 超过单文件页数上限时本地分片（用 PyMuPDF 只做切片，不做内容解析）
        chunks = self._split_if_needed(data, pdf_path)
        # (bytes, page_offset)

        # 3. 批量申请上传链接（一次请求提交所有分片）
        token = config.MINERU_API_TOKEN
        payload = {
            "files": [{"name": f"{pdf_path.stem}_p{i}.pdf", "data_id": f"{doc_id}_c{i}"}
                      for i in range(len(chunks))],
            "model_version": config.MINERU_MODEL_VERSION,
            "language": "ch",
            "enable_table": True,
        }
        r = requests.post(f"{_BASE}/file-urls/batch", headers=_headers(),
                          json=payload, timeout=60)
        r.raise_for_status()
        body = r.json()
        if body.get("code") != 0:
            raise RuntimeError(f"MinerU 申请上传链接失败: {body.get('msg')}")
        batch_id = body["data"]["batch_id"]
        urls = body["data"]["file_urls"]

        # 4. 上传各分片（官方要求 PUT 且不设 Content-Type）
        for (chunk_bytes, _), url in zip(chunks, urls):
            up = requests.put(url, data=chunk_bytes, timeout=300)
            if up.status_code not in (200, 201):
                raise RuntimeError(f"MinerU 文件上传失败: HTTP {up.status_code}")

        # 5. 轮询批量结果
        pages: list[Page] = []
        t0 = time.time()
        while True:
            time.sleep(_POLL_INTERVAL)
            rr = requests.get(f"{_BASE}/extract-results/batch/{batch_id}",
                              headers=_headers(), timeout=60)
            rr.raise_for_status()
            results = rr.json().get("data", {}).get("extract_result", [])
            states = [x.get("state") for x in results]
            if results and all(s in ("done", "failed") for s in states):
                break
            if time.time() - t0 > _POLL_TIMEOUT:
                raise RuntimeError("MinerU 云端解析超时（40 分钟），请稍后在 mineru.net 后台查看任务状态")
        for (chunk_bytes, offset), res in zip(chunks, results):
            if res.get("state") == "failed":
                raise RuntimeError(f"MinerU 解析失败: {res.get('err_msg', '未知原因')}")
            pages.extend(self._fetch_zip_pages(res["full_zip_url"], offset))

        pages.sort(key=lambda p: p.page_no)
        # 6. 落盘缓存
        cache_file.write_text(json.dumps({
            "pages": [{"page_no": p.page_no, "text": p.text,
                       "tables": [{"caption": t.caption, "rows": t.rows} for t in p.tables]}
                      for p in pages]}, ensure_ascii=False), encoding="utf-8")
        return ParsedDocument(doc_id=doc_id, doc_name=name, pages=pages)

    # ---------------- 内部工具 ----------------
    def _split_if_needed(self, data: bytes, pdf_path: Path) -> list[tuple[bytes, int]]:
        """页数 ≤ 600 直接整传；否则用 PyMuPDF 按 500 页切片（offset 为 0 基全局页码）"""
        try:
            import pymupdf as fitz
        except ImportError:
            import fitz
        src = fitz.open(stream=data, filetype="pdf")
        n = src.page_count
        if n <= 200:
            return [(data, 0)]
        chunks = []
        for start in range(0, n, _CHUNK_PAGES):
            sub = fitz.open()
            sub.insert_pdf(src, from_page=start,
                           to_page=min(start + _CHUNK_PAGES, n) - 1)
            chunks.append((sub.tobytes(), start))
            sub.close()
        src.close()
        return chunks

    def _fetch_zip_pages(self, zip_url: str, page_offset: int) -> list[Page]:
        """下载结果 zip，从 content_list.json 构建页面"""
        zr = requests.get(zip_url, timeout=300)
        zr.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(zr.content)) as zf:
            cl_name = next((n for n in zf.namelist()
                            if n.endswith("content_list.json")), None)
            if not cl_name:
                raise RuntimeError("MinerU 结果包中未找到 content_list.json")
            items = json.loads(zf.read(cl_name).decode("utf-8"))
        return _content_list_to_pages(items, page_offset)
