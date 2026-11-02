"""问题类型分类（分题型生成的前置路由）"""
from __future__ import annotations

import enum

from .. import config
from ..llm import get_llm, parse_json


class QuestionType(str, enum.Enum):
    NUMERIC = "numeric"        # 数值题：具体指标取值
    BOOLEAN = "boolean"        # 布尔题：是/否判断
    LIST = "list"              # 名单题：列举项
    COMPARE = "compare"        # 对比题：跨公司/跨期对比
    OPEN_SUMMARY = "open_summary"  # 开放总结题：全局归纳


_PROMPT = """请判断下面这个面向上市公司财报知识库的中文问题属于哪种题型，并抽取其中提到的公司名。

题型定义：
- numeric: 求某个具体数值指标（如"XX公司2024年营业收入是多少"）
- boolean: 判断正误/是否（如"XX公司毛利率是否超过30%"）
- list: 列举多项内容（如"列出XX公司前五大客户"）
- compare: 对比多家公司或多个报告期的指标（如"对比A和B公司的净利润"）
- open_summary: 全局归纳/总结/趋势类（如"总结XX公司的主要风险"）

可选公司列表：{companies}

问题：{query}

输出 JSON：{{"type": "numeric|boolean|list|compare|open_summary",
"companies": ["提到的公司名（必须在可选列表内，用问题原文表述；没提则空）"],
"metrics": ["涉及的关键指标词"]}}"""


def classify_question(query: str, available_companies: list[str]) -> dict:
    """返回 {type: QuestionType, companies: [...], metrics: [...]}"""
    llm = get_llm()
    try:
        data = llm.chat_json(_PROMPT.format(
            query=query, companies="、".join(available_companies) or "（无）"
        ), model=config.LLM_MODEL_FAST)  # 分类是轻量任务，用小模型降低延迟
        qtype = QuestionType(data.get("type", "numeric"))
    except Exception:
        # 兜底：关键词规则
        if any(w in query for w in ("对比", "比较", "和…比", "哪个更")):
            qtype = QuestionType.COMPARE
        elif any(w in query for w in ("总结", "概述", "趋势", "主要风险", "整体")):
            qtype = QuestionType.OPEN_SUMMARY
        elif any(w in query for w in ("是否", "有没有", "能否", "是否为")):
            qtype = QuestionType.BOOLEAN
        elif any(w in query for w in ("列出", "哪些", "列举", "前几大")):
            qtype = QuestionType.LIST
        else:
            qtype = QuestionType.NUMERIC
        data = {}
    companies = [c for c in data.get("companies", []) if c in available_companies]
    metrics = data.get("metrics", []) or []
    return {"type": qtype, "companies": companies, "metrics": metrics}
