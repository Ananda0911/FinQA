"""AnswerEngine：问答主链路

分类 → 公司路由 → 混合检索+Rerank+Small-to-Big → （开放题走 GraphRAG 全局摘要）
→ 分题型 Prompt → LLM 生成 → 解析【推理/答案/引用】→ 沉淀候选知识
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .. import config
from ..llm import get_llm
from ..store import get_kb
from .prompts import build_prompt
from .query_types import QuestionType, classify_question


@dataclass
class AnswerResult:
    query: str
    question_type: str
    companies: list[str]
    reasoning: str = ""
    answer: str = ""
    citations: list[dict] = field(default_factory=list)     # 引用的资料块
    contexts: list[dict] = field(default_factory=list)     # 实际送入 LLM 的资料
    graph_context_used: bool = False
    knowledge_used: list[dict] = field(default_factory=list)
    ok: bool = True
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "query": self.query, "question_type": self.question_type,
            "companies": self.companies, "reasoning": self.reasoning,
            "answer": self.answer, "citations": self.citations,
            "contexts": self.contexts, "graph_context_used": self.graph_context_used,
            "knowledge_used": self.knowledge_used, "ok": self.ok, "error": self.error,
        }


_SECTIONS = ("【推理】", "【答案】", "【引用】")


def _split_sections(text: str) -> tuple[str, str, str]:
    """把模型输出切分为 推理/答案/引用 三段"""
    def pick(head: str) -> str:
        m = re.search(re.escape(head) + r"\s*(.*?)(?=【|$)", text, flags=re.S)
        return m.group(1).strip() if m else ""
    reasoning = pick("【推理】")
    answer = pick("【答案】")
    citations = pick("【引用】")
    if not answer:  # 模型未按格式输出时，整段视为答案
        answer = text.strip()
    return reasoning, answer, citations


class AnswerEngine:
    def __init__(self):
        self.kb = get_kb()
        self.llm = get_llm()

    # ---------------- 公司路由 ----------------
    def _route(self, query: str, company: str | None) -> tuple[list[str], QuestionType, dict]:
        all_companies = self.kb.list_companies()
        if not all_companies:
            raise RuntimeError("知识库为空，请先在「文档入库」页上传财报 PDF")
        info = classify_question(query, all_companies)
        qtype: QuestionType = info["type"]
        companies = info["companies"]
        if not companies:
            companies = [company] if company and company in all_companies else all_companies
        # 对比题至少需要两家公司
        if qtype == QuestionType.COMPARE and len(companies) < 2:
            companies = all_companies
        return companies, qtype, info

    # ---------------- 主入口 ----------------
    def prepare(self, query: str, company: str | None = None,
                use_graphrag: bool = True) -> tuple[AnswerResult, list | None]:
        """问答前半段：分类 → 路由 → 检索 → 组装 Prompt。

        返回 (result, messages)：
        - messages 非 None：正常流程，可调用 chat/chat_stream 生成
        - messages 为 None：已得终态（无检索结果/检索失败），result.answer 已填好
        """
        companies, qtype, info = self._route(query, company)
        result = AnswerResult(query=query, question_type=qtype.value, companies=companies)

        # 1. 检索（含 rerank + small-to-big）
        try:
            blocks = self.kb.retrieve(query, companies)
        except Exception as e:
            result.ok, result.error = False, f"检索失败: {e}"
            return result, None
        if not blocks:
            result.answer = "N/A（知识库中未检索到相关资料）"
            result.ok = False
            result.error = "no_context"
            return result, None
        result.contexts = [dict(b) for b in blocks]

        # 2. 对话沉淀知识参考
        try:
            from ..maintenance.knowledge import search_knowledge
            refs = search_knowledge(query)
            if refs:
                result.knowledge_used = refs
        except Exception:
            pass

        # 3. 开放总结题 → GraphRAG 全局社区摘要补全
        extra_context = ""
        if use_graphrag and qtype == QuestionType.OPEN_SUMMARY:
            try:
                from ..graphrag.search import global_context
                extra_context = global_context(query, companies)
                result.graph_context_used = bool(extra_context)
            except Exception:
                extra_context = ""

        # 4. 分题型 Prompt
        messages = build_prompt(query, qtype, blocks, extra_context=extra_context)
        return result, messages

    # ---------------- 生成收尾 ----------------
    def finalize(self, result: AnswerResult, out: str) -> AnswerResult:
        """问答后半段：解析【推理/答案/引用】并提取引用溯源"""
        reasoning, answer, _ = _split_sections(out)
        result.reasoning, result.answer = reasoning, answer

        blocks = result.contexts
        used = sorted({int(i) for i in re.findall(r"\[资料(\d+)\]", answer + reasoning)})
        result.citations = [
            {k: b.get(k) for k in ("company", "doc_name", "page_no")}
            for i in used if 1 <= i <= len(blocks)
            for b in [blocks[i - 1]]
        ]
        if not result.citations:  # 模型没标引用 → 全部资料视为潜在来源
            result.citations = [
                {k: b.get(k) for k in ("company", "doc_name", "page_no")} for b in blocks
            ]
        return result

    def answer(self, query: str, company: str | None = None,
               use_graphrag: bool = True) -> AnswerResult:
        result, messages = self.prepare(query, company, use_graphrag)
        if messages is None:
            return result
        try:
            out = self.llm.chat(messages, temperature=0.1, max_tokens=3000)
        except Exception as e:
            result.ok, result.error = False, f"生成失败: {e}"
            return result
        return self.finalize(result, out)


_engine: AnswerEngine | None = None


def get_engine() -> AnswerEngine:
    global _engine
    if _engine is None:
        _engine = AnswerEngine()
    return _engine
