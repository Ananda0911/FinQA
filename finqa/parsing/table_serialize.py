"""LLM 表格序列化：把「单元格数字」还原成可检索的语义块

对跨行/跨列表格，用 LLM 将其改写为若干自然语言事实句，每句包含
行主语 + 列维度 + 数值 + 单位，避免表格问答丢数。
结果落盘缓存，同一表格不重复计费。
多表并发调用 LLM（I/O 密集），入库速度可数倍提升。
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .. import config
from ..llm import LLMClient, get_llm
from .base import ParsedDocument, TableBlock

_PROMPT = """你是一名财务数据专家。下面是一张从上市公司财报 PDF 中抽取的表格（第一行为表头）。
请把表格改写成中文事实句列表，要求：
1. 每个数据点一条句子，必须显式包含：行主语（项目/科目名）、列维度（年份/报告期/列头）、数值、单位（若表格中出现）；
2. 跨行/跨列合并单元格需还原完整语义（如"2023年及2024年"分别成句）；
3. 不遗漏任何一个数字，不要计算、不要推断、不要总结；
4. 每行一句，输出为纯文本句子列表（每行一条）。

表格标题：{caption}
表格内容：
{table}
"""

_FALLBACK_TMPL = "「{caption}」表中，{header}的数据为：{pairs}"


def _cache_key(table: TableBlock) -> str:
    raw = json.dumps([table.caption, table.rows], ensure_ascii=False)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _load_cache() -> dict:
    f = config.CACHE_DIR / "table_serial.json"
    if f.exists():
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_cache(cache: dict):
    f = config.CACHE_DIR / "table_serial.json"
    f.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")


def _naive_serialize(table: TableBlock) -> str:
    """LLM 不可用时的确定性回退：行列交叉展开"""
    if not table.rows:
        return ""
    header = table.caption or "、".join(c for c in table.rows[0] if c)
    pairs = []
    for row in table.rows[1:]:
        subject = row[0] if row else ""
        for col, val in zip(table.rows[0][1:], row[1:]):
            if str(val).strip():
                pairs.append(f"{subject}的{col}为{val}")
    return _FALLBACK_TMPL.format(caption=table.caption or header, header=header, pairs="；".join(pairs))


def _llm_serialize(table: TableBlock, llm: LLMClient) -> str:
    """单表 LLM 序列化（带重试）。全部失败返回空串，由调用方回退。"""
    prompt = _PROMPT.format(
        caption=table.caption or "（无）",
        table=table.to_markdown()[:6000],
    )
    for attempt in range(3):  # 瞬时错误（限流/网络）重试
        try:
            text = llm.chat([{"role": "user", "content": prompt}],
                            temperature=0.1, max_tokens=2048).strip()
            return text if len(text) >= 8 else ""
        except Exception:
            if attempt == 2:
                return ""
            time.sleep(2 * (attempt + 1))
    return ""


def serialize_table(table: TableBlock, llm: LLMClient | None = None) -> str:
    """单表序列化：LLM 优先，失败回退确定性展开；带磁盘缓存"""
    key = _cache_key(table)
    cache = _load_cache()
    if key in cache:
        return cache[key]
    text = _llm_serialize(table, llm or get_llm())
    if not text:
        text = _naive_serialize(table)
    else:
        cache[key] = text
        _save_cache(cache)
    return text


def serialize_tables(doc: ParsedDocument, llm: LLMClient | None = None,
                     progress=None) -> ParsedDocument:
    """将文档内所有表格并发序列化，并把序列化文本附到对应页面的 text 末尾

    - 已缓存的表格直接命中，不重复计费
    - 未缓存的表格经线程池并发调 LLM（I/O 密集，默认 6 并发）
    - LLM 成功的结果才写缓存；回退结果不缓存，下次入库自动重试
    - 每 10 张落盘一次缓存，中断不丢进度
    """
    llm = llm or get_llm()
    cache = _load_cache()
    tasks = []  # (page_idx, table, key)
    for pi, page in enumerate(doc.pages):
        for t in page.tables:
            tasks.append((pi, t, _cache_key(t)))
    n_tables = len(tasks)
    if n_tables == 0:
        return doc

    results = {key: cache[key] for _, _, key in tasks if key in cache}
    todo = [(pi, t, key) for pi, t, key in tasks if key not in cache]
    done = n_tables - len(todo)
    if done and progress:
        progress(done, n_tables)

    lock = threading.Lock()

    def _work(item):
        """worker 线程只做纯计算：LLM 调用 + 回退展开，不碰任何共享状态/界面"""
        pi, t, key = item
        text = _llm_serialize(t, llm)
        llm_ok = bool(text)
        if not llm_ok:
            try:
                text = _naive_serialize(t)  # 单表回退失败不影响整体入库
            except Exception:
                text = ""
        return key, text, llm_ok

    if todo:
        from concurrent.futures import as_completed
        with ThreadPoolExecutor(max_workers=config.TABLE_SERIAL_WORKERS) as ex:
            futs = [ex.submit(_work, it) for it in todo]
            # 结果收集/缓存/进度全部在主线程：Streamlit 非线程安全，
            # 严禁在 worker 里调用 progress（会触发 UI 组件更新导致崩溃）
            for fut in as_completed(futs):
                key, text, llm_ok = fut.result()
                with lock:
                    results[key] = text
                    # LLM 成功才进缓存；回退结果不缓存，下次入库自动重试高质量版
                    if llm_ok and key not in cache:
                        cache[key] = text
                    done += 1
                    d = done
                if progress:
                    progress(d, n_tables)
                if d % 10 == 0:
                    _save_cache(cache)
        _save_cache(cache)

    # 按页回填
    by_page: dict[int, list[str]] = {}
    for pi, t, key in tasks:
        s = results.get(key, "").strip()
        if s:
            by_page.setdefault(pi, []).append(s)
    for pi, blocks in by_page.items():
        page = doc.pages[pi]
        sep = "\n\n" if page.text else ""
        page.text = page.text + sep + "\n\n".join(blocks)
    return doc
