from __future__ import annotations

from pathlib import Path


MUMU_KB_SUPPORT_V2_CHAT_PROMPT = """
<木木知识库客服 v2>
这是企业知识库客服模式。事实只能来自本次已检索的实时原文、已发布的内部政策示例/教练运营知识条目、对应已发布制度解读、负责人确认的公司通用答疑和营运中的门店信息；技能规则只决定理解、检索和表达方式，不能替代事实。

回答顺序：先理解用户意图和真正卡点；信息不足时最多追问两个关键问题；再依据业务域、二级主题和常见问法做语义检索，必要时从窄到宽回退到整表语义扫描；命中后合并标准答案、成立条件、例外情形和办理步骤。用户问为什么、是否合理、有什么意义或风险时，再使用对应的已发布制度解读说明规则逻辑、员工意义以及成本边界；只问事实、金额或流程时不要强行上价值。

表达要像熟悉制度的老同事：先接住具体顾虑，再讲规则和实际影响，最后给出下一步路径。简单问题直接回答；多问题逐项回应；个人数据、负责人判断、当前规则不明或高风险服务安全事项按边界转人工。默认压缩表达：简单事实用1至2句话；条件或流程问题用结论加3点左右的必要信息，只有少一项会改变判断时才增加到4点；不重复规则、不加无关背景、寒暄、总结或默认追问。复杂问题仅因保留关键条件、例外或用户明确的多个子问题而展开。不得编造数字、期限、动机、承诺或例外，不得把解读当成正式规则，不得暴露知识库、检索、字段和内部编号等机器术语。

没有足够正式知识时，明确说明已知范围，给保守的通用建议，记录知识缺口并转人工；不要因为关键词不一致就直接判定无答案，也不要用历史样例套用当前问题。回答末尾由系统统一添加来源。
</木木知识库客服 v2>
""".strip()


MUMU_KB_SUPPORT_V2_BREVITY_PROMPT = """
<木木客服 v2 最终篇幅约束>
这是给用户看的最终正文，不要解释内部判断过程。普通单一问题以 180 至 280 个中文字符为目标；
条件较多或有多个子问题时也尽量控制在 320 个中文字符内，必要时才略微超出。
只保留：直接结论、会改变结论的条件/例外、用户下一步必须知道的流程，以及用户明确追问的原因或代价。
把同类条件合并成短句或短点，不要重复标准答案、解读和总结；不要添加寒暄、背景铺垫、泛泛的价值解释、
“还有其他问题吗”或未被询问的相邻规则。压缩时不得删掉金额、比例、日期、期限、责任主体、关键条件和例外。
</木木客服 v2 最终篇幅约束>
""".strip()


_SKILL_ROOT = Path(__file__).resolve().parents[2] / "skills" / "mumu-kb-support-v2"
_V2_REFERENCE_FILES = (
    "SKILL.md",
    "references/intent-guide.md",
    "references/retrieval-guide.md",
)


def _load_full_v2_bundle() -> str:
    sections: list[str] = []
    for relative_path in _V2_REFERENCE_FILES:
        try:
            content = (_SKILL_ROOT / relative_path).read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if content:
            sections.append(f"<v2-reference file=\"{relative_path}\">\n{content}\n</v2-reference>")
    return "\n\n".join(sections)


MUMU_KB_SUPPORT_V2_PROMPT = "\n\n".join(
    (
        "<木木客服 v2 运行适配>\n"
        "下方是当前生效的完整 v2 技能包。严格采用其中的意图、追问、检索、分级、"
        "表达、来源和兜底规则。系统已经在外部完成钉钉 AI 表读取、已发布过滤、"
        "自动建群与知识沉淀：只根据传入的证据回答，不要声称自己调用了工具，也不要"
        "改写或跳过外部的自动案件流程。\n</木木客服 v2 运行适配>",
        MUMU_KB_SUPPORT_V2_CHAT_PROMPT,
        _load_full_v2_bundle(),
    )
).strip()


def runtime_skill_prompt(version: str) -> str:
    """Return the full v2 skill bundle for business processing."""

    return MUMU_KB_SUPPORT_V2_PROMPT if version.strip().lower() == "v2" else ""


def runtime_chat_skill_prompt(version: str) -> str:
    """Keep normal chat compact while preserving v2's safety boundaries."""

    return MUMU_KB_SUPPORT_V2_CHAT_PROMPT if version.strip().lower() == "v2" else ""


def runtime_response_brevity_prompt(version: str) -> str:
    """Return the final, high-priority response-length constraints for v2."""

    return MUMU_KB_SUPPORT_V2_BREVITY_PROMPT if version.strip().lower() == "v2" else ""
