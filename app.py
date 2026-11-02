"""企业级财报智能问答系统 — Streamlit 入口

页面：💬 智能问答 | 📥 文档入库 | 🩺 知识库健康度 | 🔬 切片 A/B | 📚 沉淀知识
"""
from __future__ import annotations

import json
import traceback
from pathlib import Path

import streamlit as st

from finqa import config
from finqa.llm import LLMError

st.set_page_config(page_title="企业级财报智能问答", page_icon="📊", layout="wide")

# 强制显示经典常驻滚动条（Streamlit/Win11 默认悬浮式滚动条需悬停才显示，易被误认为"不能滚动"）
st.markdown("""
<style>
  /* 显式命中 Streamlit 的实际滚动容器（新版 stMain 区块） */
  section[data-testid="stMain"], [data-testid="stAppViewContainer"],
  section.main, [class*="stAppViewContainer"] {
    overflow-y: auto !important;
  }
  /* Chromium/Edge/Chrome：经典常驻滚动条，内容超高时始终可见 */
  ::-webkit-scrollbar, section[data-testid="stMain"]::-webkit-scrollbar {
    width: 12px; height: 12px; -webkit-appearance: none;
  }
  ::-webkit-scrollbar-thumb, section[data-testid="stMain"]::-webkit-scrollbar-thumb {
    background: rgba(130, 130, 130, 0.7);
    border-radius: 6px;
    border: 2px solid rgba(255, 255, 255, 0.9);
  }
  ::-webkit-scrollbar-thumb:hover { background: rgba(100, 100, 100, 0.9); }
  ::-webkit-scrollbar-track, section[data-testid="stMain"]::-webkit-scrollbar-track {
    background: rgba(240, 242, 246, 0.9); border-radius: 6px;
  }
  /* Firefox 兼容 */
  html, section[data-testid="stMain"] {
    scrollbar-width: auto;
    scrollbar-color: rgba(130, 130, 130, 0.7) rgba(240, 242, 246, 0.9);
  }
</style>
""", unsafe_allow_html=True)


# ---------------- 通用工具 ----------------
def require_api_key() -> bool:
    if not config.DASHSCOPE_API_KEY:
        st.error("未配置 DASHSCOPE_API_KEY：请复制 `.env.example` 为 `.env`，"
                 "填入阿里云百炼 API Key 后重启应用。")
        return False
    return True


def handle_error(e: Exception, context: str = ""):
    # 错误落盘：data/logs/error.log，便于事后排查（无需用户截图）
    try:
        from datetime import datetime
        log_dir = Path("data/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(log_dir / "error.log", "a", encoding="utf-8") as fh:
            fh.write(f"\n{'='*60}\n[{datetime.now():%Y-%m-%d %H:%M:%S}] {context}\n"
                     f"{type(e).__name__}: {e}\n{traceback.format_exc()}\n")
    except Exception:
        pass
    if isinstance(e, LLMError):
        st.error(f"模型调用失败：{e}")
    else:
        st.error(f"出错了：{e}")
        with st.expander("详细堆栈"):
            st.code(traceback.format_exc())


# ---------------- 侧边栏 ----------------
with st.sidebar:
    st.title("📊 财报智能问答")
    st.caption("企业级 RAG · MinerU 解析 · 混合检索 · GraphRAG")

    if not require_api_key():
        st.stop()

    from finqa.store import get_kb

    page = st.radio(
        "功能",
        ["💬 智能问答", "📥 文档入库", "🩺 知识库健康度", "🔬 切片 A/B", "📚 沉淀知识"],
        label_visibility="collapsed",
    )
    st.divider()
    companies = get_kb().list_companies()
    st.caption(f"已入库公司：{len(companies)} 家")

    with st.expander("关于本系统"):
        st.markdown(
            "- **解析**：MinerU（中文优先）→ LLM 表格序列化\n"
            "- **检索**：公司路由 · BM25+向量加权融合 · Small-to-Big 父页面回捞\n"
            "- **生成**：Rerank 精排 · 分题型 Prompt · 强制 CoT + 页码引用 · 口径不符返回 N/A\n"
            "- **进阶**：Doc2Query 扩展 · GraphRAG 全局归纳 · 健康度检查 · 切片 A/B"
        )


# ---------------- 页面：智能问答 ----------------
def render_chat():
    st.header("💬 智能问答")
    kb = get_kb()
    companies = kb.list_companies()
    if not companies:
        st.info("知识库为空，请先到「📥 文档入库」上传财报 PDF。")
        return

    col1, col2 = st.columns([3, 2])
    with col1:
        scope = st.selectbox(
            "公司范围",
            ["自动路由（含跨公司对比）"] + companies,
        )
    with col2:
        use_graph = st.toggle("GraphRAG 全局归纳（开放总结题）", value=True)

    if "chat" not in st.session_state:
        st.session_state.chat = []

    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            render_message(msg.get("payload"))

    q = st.chat_input("例如：XX公司2024年营业收入是多少？/ 对比A和B公司的净利润")
    if not q:
        return
    st.session_state.chat.append({"role": "user", "payload": {"text": q}})
    with st.chat_message("user"):
        st.markdown(q)

    from finqa.generation.answer import get_engine

    company = None if scope.startswith("自动路由") else scope
    payload = {"text": q}
    st.session_state.chat.append({"role": "assistant", "payload": payload})
    with st.chat_message("assistant"):
        streamed = False
        try:
            engine = get_engine()
            # 前半段：分类 → 检索 → 组装 Prompt（约 1-3 秒）
            with st.spinner("🔎 检索资料中…"):
                result, messages = engine.prepare(q, company=company, use_graphrag=use_graph)
            if messages is not None:
                # 后半段：流式生成（首字约 1-2 秒，逐字输出）
                from finqa.llm import get_llm
                full = st.write_stream(get_llm().chat_stream(
                    messages, temperature=0.1, max_tokens=3000))
                payload["res"] = engine.finalize(result, full).to_dict()
                streamed = True
            else:
                # 终态：N/A 或检索失败，无需生成
                payload["res"] = result.to_dict()
        except Exception as e:
            payload["error"] = str(e)
            st.error(f"出错了：{payload['error']}")
        if streamed:
            # 答案已在上方流式展示，这里只补引用溯源
            res = payload["res"]
            with st.expander(f"📎 引用与证据（{len(res['citations'])} 条）"):
                for i, c in enumerate(res["citations"], 1):
                    st.markdown(f"[资料{i}] **{c['company']}** | {c['doc_name']} | 第 {c['page_no']} 页")
                for i, ctx in enumerate(res["contexts"], 1):
                    with st.expander(f"[资料{i}] 第 {ctx['page_no']} 页原文（{ctx['doc_name']}）"):
                        st.text(ctx["page_text"][:1200] + ("…" if len(ctx["page_text"]) > 1200 else ""))
        else:
            render_message(payload)
        # 知识沉淀入口
        if payload.get("res", {}).get("ok") and payload["res"]["answer"] not in ("N/A",):
            if st.button("✔ 确认无误，沉淀为知识", key=f"save_{len(st.session_state.chat)}"):
                from finqa.maintenance.knowledge import add_knowledge
                add_knowledge(q, payload["res"]["answer"], payload["res"]["citations"])
                st.toast("已沉淀到知识库", icon="✅")
                st.rerun()


def render_message(payload):
    if payload.get("error"):
        st.error(f"出错了：{payload['error']}")
        return
    res = payload.get("res")
    if not res:
        st.markdown(payload.get("text", ""))
        return
    if not res["ok"] and not res["answer"]:
        st.error(f"回答失败：{res['error']}")
        return

    t = res["question_type"]
    badges = {
        "numeric": "`数值题`", "boolean": "`布尔题`", "list": "`名单题`",
        "compare": "`对比题`", "open_summary": "`开放总结`",
    }
    head = f"{badges.get(t, '`' + t + '`')} · 公司：{('、'.join(res['companies']) or '—')}"
    if res.get("graph_context_used"):
        head += " · 🌐 GraphRAG"
    if res.get("knowledge_used"):
        head += " · 📚 复用沉淀知识"
    st.caption(head)
    st.markdown(res["answer"])
    if res["reasoning"]:
        with st.expander("🔍 推理过程（CoT）"):
            st.markdown(res["reasoning"])
    with st.expander(f"📎 引用与证据（{len(res['citations'])} 条）"):
        for i, c in enumerate(res["citations"], 1):
            st.markdown(f"[资料{i}] **{c['company']}** | {c['doc_name']} | 第 {c['page_no']} 页")
        for i, ctx in enumerate(res["contexts"], 1):
            with st.expander(f"[资料{i}] 第 {ctx['page_no']} 页原文（{ctx['doc_name']}）"):
                st.text(ctx["page_text"][:1200] + ("…" if len(ctx["page_text"]) > 1200 else ""))


# ---------------- 页面：文档入库 ----------------
def render_ingest():
    st.header("📥 文档入库")
    st.caption("PDF 年报/研报 → MinerU 解析 → LLM 表格序列化 → 分块 → 三路索引（向量/BM25/Doc2Query）")
    kb = get_kb()

    company = st.text_input("公司名称（作为索引路由键）", placeholder="如：贵州茅台")
    files = st.file_uploader("上传财报 PDF（可多选）", type=["pdf"], accept_multiple_files=True)
    col1, col2, col3 = st.columns(3)
    backend = col1.selectbox(
        "解析后端",
        ["auto（推荐）", "mineru_cloud（云端高精度，需 Token）", "mineru（本地）", "pymupdf"],
        help="auto: 本地 MinerU → 云端 MinerU → PyMuPDF 依次回退；云端 MinerU 需在 .env 配置 MINERU_API_TOKEN",
    )
    table_serial = col2.toggle("LLM 表格序列化", value=True)
    doc2query = col3.toggle("Doc2Query 召回扩展", value=False,
                            help="为每个切片生成扩展问题，显著提升召回，消耗额外 Token")

    if st.button("开始入库", type="primary", disabled=not (company and files)):
        backend_val = "auto" if backend.startswith("auto") else backend.split()[0]
        import tempfile
        from pathlib import Path

        log = st.status(f"正在入库 {len(files)} 份文档…", expanded=False)
        try:
            for f in files:
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp.write(f.getvalue())
                    tmp_path = Path(tmp.name)
                log.write(f"#### {f.name}\n解析中…")

                # 进度节流：每个阶段每完成 10% 才写一次日志，
                # 避免 Streamlit 频繁自动滚动导致页面"拉扯"
                _last = {"key": None}

                def emit(stage, i, n):
                    key = (stage, int(100 * i / max(n, 1)) // 10)
                    if key != _last["key"]:
                        log.write(f"- {stage}: {i}/{n}")
                        _last["key"] = key

                info = kb.ingest_pdf(
                    company.strip(), tmp_path, backend=backend_val,
                    do_table_serialize=table_serial, do_doc2query=doc2query,
                    doc_name=Path(f.name).stem,  # 用原始文件名做引用溯源（临时文件名是乱码）
                    progress=emit,
                )
                log.write(f"✅ 完成：{info['pages']} 页 / {info['chunks']} 个切片")
            log.update(label=f"入库完成：{company} 共 {len(files)} 份文档", state="complete")
            st.rerun()
        except Exception as e:
            log.update(label="入库失败", state="error")
            handle_error(e, context=f"入库失败 company={company!r} files={[f.name for f in files]} "
                                    f"backend={backend_val} table_serial={table_serial} doc2query={doc2query}")

    st.divider()
    st.subheader("已入库公司")
    for c in kb.list_companies():
        s = kb.company_stats(c)
        with st.expander(f"**{c}** — {len(s['docs'])} 份文档 / {s['chunks']} 切片"
                         + (f" / Doc2Query {s['doc2query_chunks']} 切片" if s["doc2query_chunks"] else "")):
            for d in s["docs"]:
                st.markdown(f"- `{d['name']}` — {d['pages']} 页 → {d['chunks']} 切片")
            from finqa.graphrag.graph import build_graph, graph_exists
            from finqa.llm import get_llm

            st.caption("GraphRAG：实体抽取 + 社区聚类摘要，支撑全局归纳题")
            if graph_exists(c):
                st.success("图谱已构建")
            elif st.button(f"为 {c} 构建 GraphRAG 图谱", key=f"graph_{c}"):
                with st.spinner("图谱构建中（实体抽取可能需要数分钟）…"):
                    try:
                        stats = build_graph(c, get_llm())
                        st.success(f"完成：{stats['nodes']} 实体 / {stats['edges']} 关系 / "
                                   f"{stats['communities']} 社区")
                    except Exception as e:
                        handle_error(e)


# ---------------- 页面：健康度 ----------------
def render_health():
    st.header("🩺 知识库健康度")
    st.caption("关键指标缺失检查 + 同一指标数值冲突检测")
    kb = get_kb()
    companies = kb.list_companies()
    if not companies:
        st.info("知识库为空。")
        return
    st.caption(f"检查指标：{'、'.join(config.REQUIRED_INDICATORS)}")
    if st.button("运行健康检查", type="primary"):
        from finqa.maintenance.health import health_report

        progress = st.progress(0.0, text="检查中…")
        total_steps = len(companies) * len(config.REQUIRED_INDICATORS)
        state = {"done": 0}

        def emit(stage, i, n):
            state["done"] += 1
            progress.progress(min(state["done"] / total_steps, 1.0),
                              text=f"检查中… {state['done']}/{total_steps}")
        try:
            report = health_report(companies, emit)
            progress.progress(1.0, text="完成")
            for r in report:
                missing = set(r["missing"])
                st.subheader(r["company"])
                cols = st.columns(2)
                with cols[0]:
                    st.markdown("**指标覆盖**")
                    for ind in config.REQUIRED_INDICATORS:
                        if ind in missing:
                            st.markdown(f"- ⚠️ {ind} — 未检出（可能缺失或表述不同）")
                        else:
                            st.markdown(f"- ✅ {ind}")
                with cols[1]:
                    st.markdown("**数值冲突**")
                    if r["conflicts"]:
                        for c in r["conflicts"]:
                            st.markdown(
                                f"- ⚠️ **{c['indicator']}**（{c['year']}）出现多值："
                                f"{' / '.join(c['values'])}，建议人工核对第 {c['pages']} 页"
                            )
                    else:
                        st.markdown("- ✅ 未检出同一指标多值冲突")
        except Exception as e:
            handle_error(e)


# ---------------- 页面：A/B ----------------
def render_ab():
    st.header("🔬 切片 A/B 对比")
    st.caption(f"v1：{config.CHUNK_SIZE}/{config.CHUNK_OVERLAP} · "
               f"v2：{config.AB_CHUNK_SIZE}/{config.AB_CHUNK_OVERLAP} — Doc2Query 生成评测集，Hit@5 / MRR 对比")
    kb = get_kb()
    companies = kb.list_companies()
    if not companies:
        st.info("知识库为空。")
        return
    company = st.selectbox("公司", companies)
    n = st.slider("评测问题数（由切片自动生成）", 5, 50, 20)
    if st.button("运行 A/B 实验", type="primary"):
        from finqa.maintenance.ab_test import run_ab_test

        try:
            with st.status("A/B 实验运行中…", expanded=True) as status:
                def emit(stage, i, n_total):
                    st.write(f"{stage}: {i}/{n_total}")
                report = run_ab_test(company, eval_n=n, emit=emit)
                status.update(label="完成", state="complete")
            import pandas as pd

            df = pd.DataFrame([report["v1"], report["v2"]],
                              index=["v1（默认切片）", "v2（大切片）"])
            st.dataframe(df)
            st.info(f"结论：**{report['verdict']}**（以 MRR 为准）")
        except Exception as e:
            handle_error(e)


# ---------------- 页面：沉淀知识 ----------------
def render_knowledge():
    st.header("📚 沉淀知识管理")
    st.caption("问答确认后沉淀的优质知识，命中相似问题时自动注入上下文")
    from finqa.maintenance.knowledge import list_knowledge, delete_knowledge

    entries = list_knowledge()
    if not entries:
        st.info("暂无沉淀知识。在问答页点击「✔ 确认无误，沉淀为知识」即可入库。")
        return
    for e in entries:
        with st.expander(f"**{e['question']}**（{e['created_at']}）"):
            st.markdown(e["answer"])
            if e["citations"]:
                st.caption("引用：" + "；".join(
                    f"{c['company']}|{c['doc_name']}|第{c['page_no']}页" for c in e["citations"]
                ))
            if st.button("删除", key=f"del_{e['id']}"):
                delete_knowledge(e["id"])
                st.rerun()


# ---------------- 路由 ----------------
pages = {
    "💬 智能问答": render_chat,
    "📥 文档入库": render_ingest,
    "🩺 知识库健康度": render_health,
    "🔬 切片 A/B": render_ab,
    "📚 沉淀知识": render_knowledge,
}
pages[page]()
