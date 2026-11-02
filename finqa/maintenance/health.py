"""知识库健康度检查：关键指标缺失 + 同一指标数值冲突"""
from __future__ import annotations

import re

from .. import config
from ..llm import get_llm
from ..store import get_kb

_EXTRACT_PROMPT = """下面是若干财报资料片段。请抽取其中明确提到的指标数值，输出 JSON：
{{"values": [{{"indicator": "指标名", "year": "报告期或年份", "value": "数值+单位"}}]}}

问题：{indicator} 的取值。
资料片段：
{chunks}"""


def check_company_health(company: str, emit=None) -> dict:
    """返回 {company, missing: [...], conflicts: [...], checked: int}"""
    kb = get_kb()
    llm = get_llm()
    missing, conflicts = [], []
    indicators = config.REQUIRED_INDICATORS
    for i, ind in enumerate(indicators):
        emit and emit("health", i, len(indicators))
        blocks = kb.retrieve(f"{company} {ind}", company, top_k=10, rerank_top_k=5,
                              do_rerank=False, parent_page_cap=5)
        if not blocks or max(b["score"] for b in blocks) < config.HEALTH_MISSING_THRESHOLD:
            missing.append(ind)
            continue
        # 冲突检测：从命中片段抽取同一指标/同一报告期的取值
        try:
            data = llm.chat_json(_EXTRACT_PROMPT.format(
                indicator=ind,
                chunks="\n---\n".join(b.get("page_text", b.get("chunk_text", ""))[:1500] for b in blocks[:4]),
            ))
            vals = {}
            for v in data.get("values", []):
                key = (v.get("indicator", ""), v.get("year", ""))
                if not key[0]:
                    continue
                vals.setdefault(key, set()).add(str(v.get("value", "")))
            for (name, year), vs in vals.items():
                if len(vs) > 1 and ind in name:
                    conflicts.append({
                        "indicator": ind, "year": year,
                        "values": sorted(vs),
                        "pages": [b["page_no"] for b in blocks[:3]],
                    })
        except Exception:
            continue
    return {"company": company, "missing": missing, "conflicts": conflicts,
            "checked": len(indicators)}


def health_report(companies: list[str], emit=None) -> list[dict]:
    return [check_company_health(c, emit) for c in companies]
