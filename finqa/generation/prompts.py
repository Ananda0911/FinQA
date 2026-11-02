"""分题型结构化 Prompt 模板

所有模板共享的硬约束：
  - 强制 CoT：先在【推理】中核对原文，再给【答案】
  - 强制引用：答案必须标注 [资料i]，资料带公司/文档/页码溯源
  - 防编造：指标口径不等价或需推算时返回 N/A，禁止使用资料外知识
"""
from __future__ import annotations

from .query_types import QuestionType

SYSTEM = (
    "你是严谨的企业财报问答助手，服务于上市公司年报/研报知识库。"
    "你只能依据用户提供的编号资料回答，绝对禁止使用资料之外的知识，绝对禁止编造数字。"
)

_CONTEXT_TMPL = """【编号资料】（每份资料标注了来源公司、文档与页码，引用时使用 [资料i]）
{contexts}
"""

_RULE_COMMON = """硬性规则：
1. 推理过程必须逐步核对：先定位资料中的原句/原表，核对指标名称、口径、报告期、单位是否与问题完全一致；
2. 指标口径不等价（如"净利润"vs"归母净利润"未经说明）、资料未覆盖、或需要推算/换算才能得到时，答案必须是 N/A；
3. 答案中每个数字与结论都必须紧跟标注引用编号 [资料i]，缺一不可——这是验收标准，无引用的回答视为无效；
4. 不得合并不同报告期的数据，不得估算；
5. 同比/环比等趋势表述必须逐字复述资料原文（资料写"持平"就写"持平"，禁止改写成"上升/下降/增长/下滑"）；
6. 报告期以资料原文为准；资料未写明年份时不得擅自补写年份。"""

_TYPE_RULES: dict[QuestionType, str] = {
    QuestionType.NUMERIC: """题型：数值题
输出要求：答案为「数值 + 单位 + 报告期 + [资料i]」，如来源含同比/环比则按原文逐字表述；口径不一致时答 N/A 并说明差异。""",
    QuestionType.BOOLEAN: """题型：布尔题
输出要求：答案只能是「是」「否」或「N/A（资料不足）」，判断依据原句需引用并标注 [资料i]。""",
    QuestionType.LIST: """题型：名单题
输出要求：逐项列出，每项带 [资料i] 引用；资料只覆盖部分时明确"资料仅列出X项"。""",
    QuestionType.COMPARE: """题型：对比题
输出要求：用 Markdown 表格逐项对比（行=对比维度，列=对象），每个数字标注 [资料i]，缺失项填 N/A，不要强行补齐。""",
    QuestionType.OPEN_SUMMARY: """题型：开放总结题
输出要求：分点归纳，每点带 [资料i] 引用；明确说明结论覆盖的资料范围，资料未涉及的方向不要臆测。""",
}

_OUTPUT_FORMAT = """
输出格式（严格遵守）：
【推理】
<逐步核对过程，含引用>
【答案】
<按题型要求输出，数值不可用时写 N/A>
【引用】
资料i: 公司 | 文档名 | 页码
（列出实际使用到的资料）"""


def build_contexts(blocks: list[dict]) -> str:
    """blocks: [{company, doc_name, page_no, page_text}]，编号为资料1..N"""
    parts = []
    for i, b in enumerate(blocks, 1):
        src = f"{b['company']} | {b['doc_name']} | 第{b['page_no']}页"
        parts.append(f"[资料{i}]（{src}）\n{b['page_text']}")
    return "\n\n".join(parts)


def build_prompt(query: str, qtype: QuestionType, blocks: list[dict],
                 extra_context: str = "") -> list[dict]:
    """组装 messages：system + 编号资料 + 题型规则 + 硬性规则 + 问题"""
    ctx = _CONTEXT_TMPL.format(contexts=build_contexts(blocks))
    if extra_context:
        ctx += f"\n【补充全局背景】（来自知识图谱社区摘要，供归纳参考，引用时标注 [图谱]）\n{extra_context}\n"
    user = "\n".join([
        ctx,
        _TYPE_RULES[qtype],
        _RULE_COMMON,
        f"【问题】\n{query}",
        _OUTPUT_FORMAT,
    ])
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
    ]
