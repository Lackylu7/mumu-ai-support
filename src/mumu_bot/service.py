from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Protocol

from .deepseek import BUSINESS_KNOWLEDGE_REQUIRED
from .domain import (
    AnswerResult,
    ContextDecision,
    ConversationTurn,
    Evidence,
    HandoffRequest,
    QuestionScope,
    RetrievalResult,
)
from .activity_refund import ActivityRefundRepository, ActivityRefundUnderstanding
from .gaps import GapRecorder
from .memory import ConversationMemory


logger = logging.getLogger(__name__)

MULTI_QUESTION_MAX_WORKERS = 3
_ACTIVITY_REFUND_HINTS = (
    "活动", "卡项", "退卡", "退款", "退费", "退课", "购卡", "课程包",
    "没上课", "未使用", "已使用", "剩余课次", "能退", "退多少",
)


BOT_CAPABILITY_RESPONSE = (
    "我主要能帮你两类事：一是回答公司制度和业务流程，比如员工合伙制、办理步骤、"
    "适用条件和例外；二是作为普通 DeepSeek 助手，写作、翻译、分析和聊天都可以。"
    "你可以直接说具体想了解什么。只有现有知识确实无法确认的公司业务问题，我才会"
    "转给负责人处理。"
)


def _split_independent_questions(question: str) -> tuple[str, ...]:
    """Split a short newline-separated list of complete questions.

    A line break is deliberately required: prose with several rhetorical
    questions must continue through the normal single-question path.
    """

    lines = tuple(line.strip() for line in question.splitlines() if line.strip())
    if not 2 <= len(lines) <= 6:
        return (question,)
    normalized = tuple(re.sub(r"^(?:[-*•]|\d+[.、)])\s*", "", line) for line in lines)
    if all(re.search(r"[？?]\s*$", line) for line in normalized):
        return normalized
    return (question,)


def _is_bot_capability_question(question: str) -> bool:
    compact = re.sub(r"[\s，。！？!?、]+", "", question).lower()
    return bool(
        re.fullmatch(
            r"(?:木木|你)?(?:都)?(?:"
            r"能(?:回答|做|干|帮我)(?:些|点)?(?:什么|啥|哪些)(?:问题|事)?|"
            r"会(?:做|干|回答)?(?:些|点)?(?:什么|啥|哪些)(?:问题|事)?|"
            r"有(?:些|点)?(?:什么|啥|哪些)(?:功能|本事)|"
            r"(?:可以|能)问(?:木木|你)(?:些|点)?(?:什么|啥|哪些)|"
            r"(?:怎么|如何)(?:使用|用)(?:木木|你)"
            r")",
            compact,
        )
    )


def _is_unsolved_feedback(question: str) -> bool:
    """Recognize an explicit request to escalate the previous business answer."""
    compact = re.sub(r"[\s，。！？!?、：:；;。]+", "", question).lower()
    if not compact or len(compact) > 24:
        return False
    return bool(
        re.fullmatch(
            r"(?:未解决|没解决|没有解决|还没解决|仍未解决|问题没解决|问题没有解决|"
            r"答案没解决|没回答上|没有回答上|回答不对|答案不对|还是不对|不满意|"
            r"没帮上忙|没有帮上忙)",
            compact,
        )
    )


def _is_ambiguous_followup(question: str) -> bool:
    compact = re.sub(r"[\s，。！？!?、]+", "", question)
    return bool(
        re.fullmatch(
            r"(?:那|这个|那个|然后|所以|此外|另外)?(?:还)?(?:有)?(?:什么|哪些|怎么|如何)?(?:呢|呀|啊|吗)?",
            compact,
        )
        and compact
    )


def _may_be_activity_refund(
    question: str,
    history: Sequence[ConversationTurn] = (),
) -> bool:
    """Avoid an extra classifier call for clearly unrelated business topics."""

    recent = " ".join(turn.question for turn in history[-3:])
    text = f"{question} {recent}"
    return any(hint in text for hint in _ACTIVITY_REFUND_HINTS)


class KnowledgeRepository(Protocol):
    def retrieve(
        self,
        question: str,
        now: datetime,
        context: Sequence[ConversationTurn] = (),
    ) -> RetrievalResult: ...


class LanguageModel(Protocol):
    def complete(self, question: str, evidence: Sequence[Evidence]) -> str: ...

    def chat(self, question: str, history: Sequence[ConversationTurn]) -> str: ...

    def resolve_context(
        self,
        question: str,
        history: Sequence[ConversationTurn],
    ) -> ContextDecision: ...

    def route_sources(
        self,
        question: str,
        history: Sequence[ConversationTurn] = (),
    ) -> object: ...

    def classify_activity_refund(
        self,
        question: str,
        history: Sequence[ConversationTurn] = (),
    ) -> ActivityRefundUnderstanding: ...


def _redact_question(question: str) -> str:
    compact = " ".join(question.strip().split())
    compact = re.sub(r"(?<!\d)1\d{10}(?!\d)", "***********", compact)
    compact = re.sub(
        r"(?<!\d)\d{17}[0-9Xx](?!\d)",
        "******************",
        compact,
    )
    return compact[:200]


def _numbers(text: str) -> set[str]:
    normalized_text = re.sub(r"(?<=\d)\s+(?=(?:万|千|百|[Kk])(?:\D|$))", "", text)
    normalized_text = re.sub(r"(?<=\d)[,，](?=\d{3}(?:\D|$))", "", normalized_text)
    without_evidence_ids = re.sub(
        r"(?<![A-Za-z0-9])EP\d+-(?:K|S)\d+(?![A-Za-z0-9])",
        "",
        normalized_text,
        flags=re.IGNORECASE,
    )
    without_list_markers = re.sub(
        r"(?m)^[ \t]*\d+(?:[.)）]|[、．])(?!\d)[ \t]*",
        "",
        without_evidence_ids,
    )
    without_source_versions = re.sub(
        r"(?:(?:员工)?合伙制|制度|版本)\s*[Vv]?\d+(?:\.\d+)+",
        "",
        without_list_markers,
    )
    numbers: set[str] = set()
    scale_factors = {
        "": Decimal(1),
        "百": Decimal(100),
        "千": Decimal(1000),
        "万": Decimal(10000),
        "k": Decimal(1000),
        "K": Decimal(1000),
    }
    for number_text, scale, percent_sign in re.findall(
        r"(\d+(?:\.\d+)?)(万|千|百|[Kk])?(%?)",
        without_source_versions,
    ):
        if percent_sign:
            numbers.add(f"{number_text}%")
            continue
        try:
            numbers.add(_formatted_number(Decimal(number_text) * scale_factors[scale]))
        except InvalidOperation:
            continue
    return numbers


def _source_metadata_numbers(text: str) -> set[str]:
    """Allow a source title's compact YYMMDD date to be cited naturally."""
    numbers = _numbers(text)
    for year, month, day in re.findall(r"(?<!\d)(\d{2})(\d{2})(\d{2})(?!\d)", text):
        month_value = int(month)
        day_value = int(day)
        if 1 <= month_value <= 12 and 1 <= day_value <= 31:
            numbers.update(
                {
                    str(2000 + int(year)),
                    str(month_value),
                    str(day_value),
                }
            )
    return numbers


def _formatted_number(value: Decimal) -> str:
    formatted = format(value.normalize(), "f")
    if "." in formatted:
        formatted = formatted.rstrip("0").rstrip(".")
    return formatted or "0"


def _derived_numbers(text: str) -> set[str]:
    raw_numbers = re.findall(
        r"(\d+(?:\.\d+)?)(万|千|百)?(%?)",
        text,
    )
    raw_amounts: list[Decimal] = []
    scaled_amounts: list[Decimal] = []
    percentages: list[Decimal] = []
    scale_factors = {
        "": Decimal(1),
        "百": Decimal(100),
        "千": Decimal(1000),
        "万": Decimal(10000),
    }
    for number_text, scale, percent_sign in raw_numbers:
        try:
            number = Decimal(number_text)
            if percent_sign:
                percentages.append(number / Decimal(100))
            else:
                raw_amounts.append(number)
                scaled_amounts.append(number * scale_factors[scale])
        except InvalidOperation:
            continue

    derived: set[str] = set()
    for match in re.finditer(
        r"前(\d+)个?(?:月|天|日|年)[^。；\n]{0,60}(?:之后|以后|期满后)",
        text,
    ):
        derived.add(str(int(match.group(1)) + 1))
    # Unit conversion is itself a valid derivation (for example, 1万元 -> 10000元).
    derived.update(_formatted_number(amount) for amount in scaled_amounts)
    for amounts in (raw_amounts, scaled_amounts):
        calculation_amounts = amounts
        if amounts and max(amounts) >= 1000:
            calculation_amounts = [amount for amount in amounts if amount >= 100]
        for amount in calculation_amounts:
            for percentage in percentages:
                percentage_amount = amount * percentage
                derived.add(_formatted_number(percentage_amount))
                derived.add(_formatted_number(amount - percentage_amount))
        for left_index, left in enumerate(calculation_amounts):
            for right in calculation_amounts[left_index + 1 :]:
                derived.add(_formatted_number(left + right))
                derived.add(_formatted_number(abs(left - right)))
        for window_size in range(3, min(4, len(calculation_amounts)) + 1):
            for start in range(len(calculation_amounts) - window_size + 1):
                window = calculation_amounts[start : start + window_size]
                derived.add(_formatted_number(sum(window, Decimal(0))))
    return derived


def _has_conflict(evidence: Sequence[Evidence]) -> bool:
    """Only flag competing sources, not complementary sections of one document.

    A source guide is commonly ingested as one record per page or section.  Those
    records naturally have the same intent but different text, and must be
    combined for a complete answer rather than treated as a contradiction.
    """
    answers_by_intent: dict[str, dict[tuple[str, str], set[str]]] = {}
    for item in evidence:
        normalized = re.sub(r"\s+", "", item.answer)
        source_revision = (item.source_id.strip(), item.version.strip())
        answers_by_intent.setdefault(item.intent, {}).setdefault(
            source_revision,
            set(),
        ).add(normalized)
    return any(
        len({answer for answers in revisions.values() for answer in answers}) > 1
        and len(revisions) > 1
        for revisions in answers_by_intent.values()
    )


def _fact_fragments(text: str) -> tuple[str, ...]:
    return tuple(
        compact
        for fragment in re.split(r"[。；;！？!?\n]+", text)
        if (compact := re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "", fragment))
    )


def _bigram_coverage(fragment: str, answer: str) -> float:
    if len(fragment) < 2:
        return float(fragment in answer)
    fragment_bigrams = {fragment[index : index + 2] for index in range(len(fragment) - 1)}
    answer_bigrams = {answer[index : index + 2] for index in range(len(answer) - 1)}
    return len(fragment_bigrams & answer_bigrams) / len(fragment_bigrams)


def _rule_is_covered_by_answer(rule: str, answer: str) -> bool:
    compact_answer = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "", answer)
    for fragment in _fact_fragments(rule):
        if fragment in compact_answer:
            continue
        if _numbers(fragment) - _numbers(compact_answer):
            return False
        if _bigram_coverage(fragment, compact_answer) < 0.55:
            return False
    return True


def _missing_rule_fragments(rule: str, answer: str) -> tuple[str, ...]:
    compact_answer = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "", answer)
    missing: list[str] = []
    for raw_fragment in re.split(r"[。；;！？!?\n]+", rule):
        raw_fragment = raw_fragment.strip()
        fragment = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "", raw_fragment)
        if not fragment:
            continue
        if fragment in compact_answer:
            continue
        if _numbers(fragment) - _numbers(compact_answer):
            missing.append(raw_fragment)
            continue
        if _bigram_coverage(fragment, compact_answer) < 0.55:
            missing.append(raw_fragment)
    return tuple(missing)


def _can_retry_insufficient(
    question: str,
    evidence: Sequence[Evidence] = (),
) -> bool:
    compact = re.sub(r"\s+", "", question)
    calculation_question = (
        len(_numbers(compact)) >= 2
        and bool(re.search(r"累计|是否达到|怎么算|怎么测算|剩余|分别", compact))
    )
    safe_refusal_question = "行业常识" in compact and "补全" in compact
    return calculation_question or safe_refusal_question or bool(evidence)


def _asks_for_interpretation(question: str) -> bool:
    compact = re.sub(r"\s+", "", question)
    return bool(
        re.search(
            r"为什么|为何|原因|意义|好处|保护|合理|公平|凭什么|"
            r"这样规定|这么规定|目的|有用|坑不坑|是不是坑|不理解|想不通|"
            r"有利的地方|真正有利|需要承担的风险|收益和风险|好处和风险|"
            r"(?:这)?不就是.*(?:扣|卡|限制|赶|逼|为难)|"
            r"是不是(?:在)?(?:扣|卡|限制|赶|逼|为难)|"
            r"为什么非要|值不值|划不划算",
            compact,
        )
    )


def _asks_for_calculation(question: str) -> bool:
    compact = re.sub(r"\s+", "", question)
    return bool(
        re.search(
            r"计算|测算|怎么算|算多少|分别多少|多少(?:月)?分红|对应(?:多少|几)",
            compact,
        )
    )


def _requests_common_sense_completion(question: str) -> bool:
    compact = re.sub(r"\s+", "", question)
    return bool(re.search(r"(?:行业)?常识.*(?:补全|补充)|按.*常识.*(?:补全|补充)", compact))


def _missing_explicit_dimensions(question: str, answer: str) -> tuple[str, ...]:
    compact_question = re.sub(r"\s+", "", question)
    compact_answer = re.sub(r"\s+", "", answer)
    dimensions = (
        (
            "对员工有利的地方或收益",
            r"有利|好处|收益|价值|保护|获得",
            r"有利|好处|收益|分红|价值|保护|获得|权利|机会",
        ),
        (
            "需要承担的风险或代价",
            r"风险|代价|成本|承担|不利|损失",
            r"风险|代价|成本|承担|亏损|损失|减少|投入|责任|负债",
        ),
    )
    return tuple(
        label
        for label, question_pattern, answer_pattern in dimensions
        if re.search(question_pattern, compact_question)
        and not re.search(answer_pattern, compact_answer)
    )


def _reviewed_fact_for_dimension(
    dimension: str,
    evidence: Sequence[Evidence],
) -> str:
    patterns = (
        (r"亏损|损失|负债|补差", 4),
        (r"承担|责任|退股|投入", 3),
        (r"风险金|共同存入|利润先存", 2),
    )
    if "有利" in dimension or "收益" in dimension:
        patterns = (
            (r"分红|收益|利润分配", 4),
            (r"职业路径|发展|晋升|权利", 3),
        )
    best_answer = ""
    best_score = 0
    for item in evidence:
        searchable = "\n".join(
            (item.answer, *item.conditions, *item.exceptions, *item.steps)
        )
        score = sum(weight for pattern, weight in patterns if re.search(pattern, searchable))
        if score > best_score:
            best_answer = item.answer
            best_score = score
    return best_answer


def _source_display_name(item: Evidence) -> str:
    source_names = {
        "POLICY-S001": "内部政策示例",
    }
    if item.source_id in source_names:
        return source_names[item.source_id]
    if item.source_id.startswith("MUMU-CHAT-"):
        return "公司内部已确认答疑"
    return item.source_id.strip() or item.intent.strip() or "公司内部知识"


def _source_citation(evidence: Sequence[Evidence], *, v2: bool = False) -> str:
    del v2
    citations: list[str] = []
    for item in evidence:
        source_name = _source_display_name(item)
        section = item.source_section.strip()
        version = item.version.strip()
        if section.endswith("实时读取"):
            library_name = section.removesuffix("实时读取").strip() or "企业知识库"
            citation = f"企业知识库 > {library_name} > 《{source_name}》 > 实时读取"
        elif item.source_id == "STORE-S001":
            citation = "木木 AI 表 > 门店信息宽表 > 门店基础信息"
        elif item.source_id.startswith("MUMU-CHAT-"):
            citation = "木木 AI 表 > 知识条目 > 公司内部已确认答疑"
            if section:
                citation += f" > {section}"
        else:
            citation = f"木木 AI 表 > 知识条目 > 《{source_name}》"
            if section:
                citation += f" > {section}"
            if version and version not in {"directory", "实时读取"}:
                citation += f"（版本：{version}）"
        if item.source_url.strip():
            # Session-webhook replies use DingTalk text messages.  DingTalk
            # recognizes a literal HTTPS URL as a clickable link, while this
            # keeps existing @-mention behavior intact.
            citation += f"（链接：{item.source_url.strip()}）"
        if citation not in citations:
            citations.append(citation)
    return "\n".join(f"- {citation}" for citation in citations[:3])


def _remove_model_source_lines(text: str) -> str:
    """Keep source attribution in one service-owned footer only."""
    lines = text.strip().splitlines()
    cleaned = [
        line
        for line in lines
        if not re.match(r"^\s*回答内容来源\s*[:：]", line)
    ]
    return "\n".join(cleaned).strip()


def _confirmed_evidence_result(
    evidence: Sequence[Evidence],
    *,
    v2: bool = False,
) -> AnswerResult:
    sections: list[str] = []
    answers = tuple(dict.fromkeys(item.answer.strip() for item in evidence if item.answer.strip()))
    sections.extend(answers)

    labelled_values = (
        ("适用条件", tuple(value for item in evidence for value in item.conditions if value.strip())),
        ("例外情形", tuple(value for item in evidence for value in item.exceptions if value.strip())),
        ("办理步骤", tuple(value for item in evidence for value in item.steps if value.strip())),
    )
    for label, values in labelled_values:
        unique_values = tuple(dict.fromkeys(values))
        if unique_values:
            sections.append(f"{label}：" + "；".join(unique_values))

    citations = _source_citation(evidence, v2=v2)
    text = "\n\n".join(sections)
    if citations:
        label = "回答内容来源" if v2 else "依据"
        text = f"{text}\n\n{label}：\n{citations}"
    logger.warning(
        "business_model_safe_fallback evidence_ids=%s",
        ",".join(item.knowledge_id for item in evidence),
    )
    return AnswerResult(text=text, evidence=tuple(evidence))


class SupportService:
    def __init__(
        self,
        *,
        repository: KnowledgeRepository,
        model: LanguageModel,
        gap_recorder: GapRecorder,
        operator_user_id: str,
        memory: ConversationMemory | None = None,
        activity_refund_repository: ActivityRefundRepository | None = None,
    ) -> None:
        self.repository = repository
        self.model = model
        self.gap_recorder = gap_recorder
        self.operator_user_id = operator_user_id
        self.memory = memory or ConversationMemory()
        self.activity_refund_repository = activity_refund_repository

    def _uses_v2_skill(self) -> bool:
        return str(getattr(self.model, "skill_version", "v1")).lower() == "v2"

    def _route_source_mode(
        self,
        question: str,
        history: Sequence[ConversationTurn],
    ) -> tuple[str | None, tuple[str, ...], str]:
        router = getattr(self.model, "route_sources", None)
        if not callable(router):
            return None, (), "single"
        try:
            decision = router(question, history)
        except Exception as error:
            logger.warning("source_route_failed error_type=%s", type(error).__name__)
            return "both", (), "single"
        live_sources = tuple(getattr(decision, "live_sources", ()) or ())
        live_search_policy = getattr(decision, "live_search_policy", "single")
        if live_search_policy not in {"single", "cross_domain", "fallback"}:
            live_search_policy = "fallback" if len(live_sources) > 1 else "single"
        mode = getattr(decision, "source_mode", decision)
        if mode not in {"realtime", "aitable", "both", "general"}:
            logger.warning("source_route_invalid mode=%r", mode)
            return "both", live_sources, live_search_policy
        return mode, live_sources, live_search_policy

    def _needs_supplemental_retrieval(
        self,
        question: str,
        evidence: Sequence[Evidence],
    ) -> bool:
        """Check whether the first source can cover the user's actual question.

        This is deliberately a coverage check instead of a lexical threshold:
        a live document may contain one related sentence while still missing a
        condition, exception, process, or another part of a compound question.
        A failed check must not make an otherwise usable answer unavailable.
        """

        if not evidence:
            return True
        analyze = getattr(self.model, "analyze_evidence", None)
        if not callable(analyze):
            return False
        try:
            assessment = analyze(question, evidence)
        except Exception as error:
            logger.warning(
                "supplemental_retrieval_coverage_check_failed error_type=%s",
                type(error).__name__,
            )
            return False
        return (
            not bool(getattr(assessment, "answerable", False))
            or bool(getattr(assessment, "handoff_recommended", False))
        )

    @staticmethod
    def _merge_evidence(
        *groups: Sequence[Evidence],
    ) -> tuple[Evidence, ...]:
        merged: list[Evidence] = []
        seen_ids: set[str] = set()
        for group in groups:
            for item in group:
                if item.knowledge_id not in seen_ids:
                    merged.append(item)
                    seen_ids.add(item.knowledge_id)
        return tuple(merged)

    def _supplemental_retrieval(
        self,
        *,
        question: str,
        now: datetime,
        context: Sequence[ConversationTurn],
        source_mode: str | None,
        live_sources: Sequence[str],
    ) -> RetrievalResult | None:
        """Fetch only the remaining candidate sources plus the AI table.

        The first live source has already been searched.  Remaining sources are
        therefore passed as a fallback chain, never as an all-library scan.
        """

        if source_mode not in {"realtime", "aitable", "both"}:
            return None
        remaining_sources = (
            tuple(live_sources[1:])
            if source_mode in {"realtime", "both"}
            else tuple(live_sources)
        )
        if source_mode in {"aitable", "both"} and not remaining_sources:
            return None
        return self.repository.retrieve(
            question,
            now,
            context,
            source_mode="both",
            live_sources=remaining_sources,
            live_search_policy="fallback",
        )

    def answer(
        self,
        user_id: str,
        question: str,
        now: datetime,
        conversation_id: str = "",
        company_name: str = "",
    ) -> AnswerResult:
        questions = _split_independent_questions(question)
        if len(questions) > 1:
            return self._answer_independent_questions(
                user_id=user_id,
                original_question=question,
                questions=questions,
                now=now,
                conversation_id=conversation_id,
                company_name=company_name,
            )
        return self._answer_one(
            user_id,
            question,
            now,
            conversation_id,
            company_name,
        )

    def _answer_independent_questions(
        self,
        *,
        user_id: str,
        original_question: str,
        questions: Sequence[str],
        now: datetime,
        conversation_id: str,
        company_name: str,
    ) -> AnswerResult:
        """Answer a compact list independently instead of forcing one intent."""

        batch_conversation_id = f"{conversation_id or 'direct'}:batch:{now.timestamp()}"

        def answer_item(index_and_question: tuple[int, str]) -> AnswerResult:
            index, item = index_and_question
            return self._answer_one(
                user_id,
                item,
                now,
                f"{batch_conversation_id}:{index}",
                company_name,
            )

        with ThreadPoolExecutor(
            max_workers=min(MULTI_QUESTION_MAX_WORKERS, len(questions)),
            thread_name_prefix="mumu-multi-question",
        ) as executor:
            results = list(executor.map(answer_item, enumerate(questions, start=1)))
        sections = [
            f"{index}. {item}\n{result.text}"
            for index, (item, result) in enumerate(zip(questions, results), start=1)
        ]
        evidence = tuple(
            item
            for result in results
            for item in result.evidence
        )
        handoffs = tuple(result.handoff for result in results if result.handoff)
        session_key = f"{conversation_id or 'direct'}:{user_id}"
        if handoffs:
            unresolved = tuple(
                item for item, result in zip(questions, results) if result.handoff
            )
            result = AnswerResult(
                text="\n\n".join(sections),
                evidence=evidence,
                handoff=HandoffRequest(
                    assignee_user_id=self.operator_user_id,
                    reason="多项问题中存在需要人工确认的内容",
                    question_summary=_redact_question("；".join(unresolved)),
                    evidence_ids=tuple(
                        dict.fromkeys(
                            evidence_item.knowledge_id for evidence_item in evidence
                        )
                    ),
                    open_case=True,
                ),
            )
            self._remember(
                session_key,
                original_question,
                result,
                QuestionScope.BUSINESS_GAP,
                now,
            )
            return result

        result = AnswerResult(text="\n\n".join(sections), evidence=evidence)
        self._remember(
            session_key,
            original_question,
            result,
            QuestionScope.BUSINESS_MATCHED,
            now,
        )
        return result

    def _answer_one(
        self,
        user_id: str,
        question: str,
        now: datetime,
        conversation_id: str = "",
        company_name: str = "",
    ) -> AnswerResult:
        session_key = f"{conversation_id or 'direct'}:{user_id}"
        history = self.memory.history(session_key, now)
        if _is_bot_capability_question(question):
            result = AnswerResult(text=BOT_CAPABILITY_RESPONSE)
            self._remember(
                session_key,
                question,
                result,
                QuestionScope.GENERAL_CHAT,
                now,
            )
            return result
        context_decision = ContextDecision(False, question)
        # A short, explicit dissatisfaction reply is a deterministic command,
        # not a new knowledge question.  This avoids relying on the LLM to
        # classify "未解决" and prevents accidental retrieval/duplicate cases.
        if _is_unsolved_feedback(question):
            previous_business_turn = next(
                (
                    turn
                    for turn in reversed(history)
                    if turn.scope
                    in {QuestionScope.BUSINESS_MATCHED, QuestionScope.BUSINESS_GAP}
                ),
                None,
            )
            if previous_business_turn is not None:
                if any(
                    marker in previous_business_turn.answer
                    for marker in ("转给人工负责人", "已创建业务答疑群", "自动建群")
                ):
                    return AnswerResult(
                        text="这个问题已经转给人工负责人处理了，请在答疑群里继续补充具体情况。"
                    )
                original_question = _redact_question(previous_business_turn.question)
                result = AnswerResult(
                    text=(
                        f"好的，已把刚才关于“{original_question}”的问题"
                        "转给人工负责人确认。"
                    ),
                    handoff=HandoffRequest(
                    assignee_user_id=self.operator_user_id,
                        reason="提问人反馈上一轮回答未解决问题",
                        question_summary=original_question,
                        open_case=True,
                        context=history,
                    ),
                )
                self._remember(
                    session_key,
                    question,
                    result,
                    QuestionScope.BUSINESS_GAP,
                    now,
                )
                return result
            return AnswerResult(text="请先针对上一条具体问题回复“未解决”，我再帮你转人工处理。")
        if history:
            try:
                context_decision = self.model.resolve_context(question, history)
            except Exception:
                context_decision = ContextDecision(False, question)
        if context_decision.interaction_intent == "handoff_confirm":
            previous_business_turn = next(
                (
                    turn
                    for turn in reversed(history)
                    if turn.scope
                    in {QuestionScope.BUSINESS_MATCHED, QuestionScope.BUSINESS_GAP}
                ),
                None,
            )
            if previous_business_turn is not None:
                original_question = _redact_question(previous_business_turn.question)
                result = AnswerResult(
                    text=(
                        f"好的，已把刚才关于“{original_question}”的问题"
                        "转给人工负责人确认。"
                    ),
                    handoff=HandoffRequest(
                    assignee_user_id=self.operator_user_id,
                        reason="用户要求人工确认上一轮业务问题",
                        question_summary=original_question,
                        open_case=True,
                        context=history,
                    ),
                )
                self._remember(
                    session_key,
                    question,
                    result,
                    QuestionScope.BUSINESS_GAP,
                    now,
                )
                return result
        relevant_history = history if context_decision.use_history else ()
        standalone_question = context_decision.standalone_question or question
        if _is_ambiguous_followup(question) and not relevant_history:
            return AnswerResult(
                text="你是接着前面哪个问题问的？带上一个关键词，我就能继续帮你看。"
            )
        source_mode, live_sources, live_search_policy = self._route_source_mode(
            standalone_question,
            relevant_history,
        )
        if source_mode == "general":
            try:
                chat_answer = self.model.chat(question, relevant_history)
            except Exception:
                return AnswerResult(text="普通聊天服务暂时不可用，请稍后重试。")
            if BUSINESS_KNOWLEDGE_REQUIRED in chat_answer:
                redacted = _redact_question(question)
                self.gap_recorder.record(redacted, "普通聊天识别为公司业务", now)
                result = self._handoff(
                    redacted,
                    "公司业务知识不足",
                    (),
                    open_case=True,
                    context=history,
                )
                self._remember(
                    session_key,
                    question,
                    result,
                    QuestionScope.BUSINESS_GAP,
                    now,
                )
                return result
            result = AnswerResult(text=chat_answer)
            self._remember(
                session_key,
                question,
                result,
                QuestionScope.GENERAL_CHAT,
                now,
            )
            return result
        if (
            self.activity_refund_repository is not None
            and _may_be_activity_refund(standalone_question, relevant_history)
        ):
            activity_intent: bool | None = None
            activity_facts: dict[str, object] | None = None
            classifier = getattr(self.model, "classify_activity_refund", None)
            if callable(classifier):
                try:
                    classified = classifier(standalone_question, relevant_history)
                    if isinstance(classified, ActivityRefundUnderstanding):
                        activity_intent = classified.is_activity_refund
                        activity_facts = classified.facts.as_mapping()
                    elif isinstance(classified, bool):
                        activity_intent = classified
                except Exception as error:
                    logger.warning(
                        "activity_refund_intent_classification_failed error_type=%s",
                        type(error).__name__,
                    )
            activity_result = self.activity_refund_repository.answer(
                user_id,
                standalone_question,
                relevant_history,
                company_name,
                activity_intent,
                activity_facts,
            )
            if activity_result is not None:
                result = AnswerResult(
                    text=activity_result.text,
                    evidence=activity_result.evidence,
                )
                self._remember(
                    session_key,
                    question,
                    result,
                    QuestionScope.BUSINESS_MATCHED,
                    now,
                )
                return result
        retrieve_kwargs = {}
        if source_mode is not None:
            retrieve_kwargs["source_mode"] = source_mode
            retrieve_kwargs["live_sources"] = live_sources
            retrieve_kwargs["live_search_policy"] = live_search_policy
        retrieval = self.repository.retrieve(
            standalone_question,
            now,
            relevant_history,
            **retrieve_kwargs,
        )

        if retrieval.scope is QuestionScope.TEMPORARY_ERROR:
            if retrieval.evidence:
                result = self._handoff(
                    standalone_question,
                    "已检索到相关资料，但语义交叉核对暂时不可用",
                    retrieval.evidence,
                    open_case=True,
                    context=history,
                    notice="木木已找到相关资料，但暂时无法完成交叉核对。为保证准确，已转人工负责人确认。",
                )
                self._remember(
                    session_key,
                    question,
                    result,
                    QuestionScope.BUSINESS_GAP,
                    now,
                )
                return result
            return AnswerResult(text="系统暂时繁忙，请稍后重试。")

        if retrieval.scope is QuestionScope.GENERAL_CHAT:
            try:
                chat_answer = self.model.chat(question, relevant_history)
            except Exception:
                return AnswerResult(text="普通聊天服务暂时不可用，请稍后重试。")
            if BUSINESS_KNOWLEDGE_REQUIRED in chat_answer:
                redacted = _redact_question(question)
                self.gap_recorder.record(redacted, "普通聊天识别为公司业务", now)
                result = self._handoff(
                    redacted,
                    "公司业务知识不足",
                    (),
                    open_case=True,
                    context=history,
                )
                self._remember(
                    session_key,
                    question,
                    result,
                    QuestionScope.BUSINESS_GAP,
                    now,
                )
                return result
            result = AnswerResult(text=chat_answer)
            self._remember(
                session_key,
                question,
                result,
                QuestionScope.GENERAL_CHAT,
                now,
            )
            return result

        evidence = tuple(retrieval.evidence)
        should_supplement = (
            source_mode == "realtime"
            or bool(live_sources[1:])
            or (source_mode == "aitable" and bool(live_sources))
        )
        if should_supplement and self._needs_supplemental_retrieval(
            standalone_question,
            evidence,
        ):
            supplemental = self._supplemental_retrieval(
                question=standalone_question,
                now=now,
                context=relevant_history,
                source_mode=source_mode,
                live_sources=live_sources,
            )
            if supplemental is not None:
                if supplemental.scope is QuestionScope.TEMPORARY_ERROR and not evidence:
                    if supplemental.evidence:
                        result = self._handoff(
                            standalone_question,
                            "已检索到相关资料，但语义交叉核对暂时不可用",
                            supplemental.evidence,
                            open_case=True,
                            context=history,
                            notice="木木已找到相关资料，但暂时无法完成交叉核对。为保证准确，已转人工负责人确认。",
                        )
                        self._remember(
                            session_key,
                            question,
                            result,
                            QuestionScope.BUSINESS_GAP,
                            now,
                        )
                        return result
                    return AnswerResult(text="系统暂时繁忙，请稍后重试。")
                evidence = self._merge_evidence(evidence, supplemental.evidence)
        if not evidence:
            redacted = _redact_question(standalone_question)
            self.gap_recorder.record(redacted, "知识库无匹配", now)
            result = self._handoff(
                redacted,
                "知识库无匹配",
                evidence,
                open_case=True,
                context=history,
            )
            self._remember(
                session_key,
                question,
                result,
                QuestionScope.BUSINESS_GAP,
                now,
            )
            return result

        result = self._answer_business(standalone_question, evidence)
        self._remember(
            session_key,
            question,
            result,
            QuestionScope.BUSINESS_MATCHED,
            now,
        )
        return result

    def _answer_business(
        self,
        question: str,
        evidence: Sequence[Evidence],
    ) -> AnswerResult:
        if _has_conflict(evidence):
            return self._handoff(
                question,
                "知识证据冲突",
                evidence,
                open_case=True,
            )
        model_answer = self._complete_business(question, evidence)
        if model_answer is None:
            return _confirmed_evidence_result(evidence, v2=self._uses_v2_skill())
        if _requests_common_sense_completion(question) and "不能按行业常识补全公司规则" not in model_answer:
            model_answer = f"不能按行业常识补全公司规则；{model_answer}"
        missing_dimensions = _missing_explicit_dimensions(question, model_answer)
        if missing_dimensions:
            retry_question = (
                f"{question}\n\n补充要求：请正面逐项回答"
                f"{'、'.join(missing_dimensions)}；只能使用给定证据，"
                "不要扩展到无关规则。"
            )
            retry_answer = self._complete_business(retry_question, evidence)
            if retry_answer is None:
                return _confirmed_evidence_result(evidence, v2=self._uses_v2_skill())
            if len(_missing_explicit_dimensions(question, retry_answer)) < len(
                missing_dimensions
            ):
                model_answer = retry_answer
        remaining_dimensions = _missing_explicit_dimensions(question, model_answer)
        for dimension in remaining_dimensions:
            reviewed_fact = _reviewed_fact_for_dimension(dimension, evidence)
            if reviewed_fact:
                model_answer = f"{model_answer}\n\n{dimension}：{reviewed_fact}"
        if "需要人工确认" in model_answer and _can_retry_insufficient(
            question,
            evidence,
        ):
            model_answer = self._complete_business(question, evidence)
            if model_answer is None:
                return _confirmed_evidence_result(evidence, v2=self._uses_v2_skill())
        if "需要人工确认" in model_answer:
            # 已检索到正式知识时，优先把可核实的规则交给用户，而非重复建群。
            return _confirmed_evidence_result(evidence, v2=self._uses_v2_skill())
        evidence_values: list[str] = []
        for item in evidence:
            evidence_values.extend(
                (
                    item.answer,
                    *item.conditions,
                    *item.exceptions,
                    *item.steps,
                    item.source_id,
                    item.source_section,
                    item.version,
                    item.answer_boundary,
                )
            )
            for interpretation in item.interpretations:
                evidence_values.extend(
                    (
                        interpretation.employee_concern,
                        interpretation.rationale,
                        interpretation.employee_value,
                        interpretation.employee_cost,
                        interpretation.recommended_response,
                        *interpretation.boundaries,
                    )
                )
        evidence_text = "\n".join(evidence_values)
        calculation_context = f"{question}\n{evidence_text}"
        direct_numbers = _numbers(evidence_text) | _numbers(question)
        source_metadata = "\n".join(
            value
            for item in evidence
            for value in (item.source_id, item.source_section, item.version)
        )
        direct_numbers.update(_source_metadata_numbers(source_metadata))
        derived_numbers = _derived_numbers(calculation_context)
        model_numbers = _numbers(model_answer)
        if model_numbers - (direct_numbers | derived_numbers):
            return self._handoff(
                question,
                "模型答案包含证据外数字",
                evidence,
                open_case=True,
            )

        v2 = self._uses_v2_skill()
        if (
            not v2
            and not _asks_for_calculation(question)
            and not _asks_for_interpretation(question)
        ):
            missing_rules = [
                fragment
                for item in evidence
                if item.answer
                for fragment in _missing_rule_fragments(
                    item.answer,
                    f"{question}\n{model_answer}",
                )
            ]
            if missing_rules:
                model_answer = "\n\n".join(
                    dict.fromkeys((model_answer, *missing_rules))
                )

        model_answer = _remove_model_source_lines(model_answer)
        citations = _source_citation(evidence, v2=v2)
        label = "回答内容来源" if v2 else "依据"
        return AnswerResult(
            text=f"{model_answer}\n\n{label}：\n{citations}",
            evidence=tuple(evidence),
        )

    def _complete_business(
        self,
        question: str,
        evidence: Sequence[Evidence],
    ) -> str | None:
        try:
            return self.model.complete(question, evidence)
        except Exception as error:
            logger.warning("business_model_failed error_type=%s", type(error).__name__)
            return None

    def _remember(
        self,
        session_key: str,
        question: str,
        result: AnswerResult,
        scope: QuestionScope,
        now: datetime,
    ) -> None:
        self.memory.append(
            session_key,
            ConversationTurn(
                question=question,
                answer=result.text,
                scope=scope,
                created_at=now,
            ),
        )

    def _handoff(
        self,
        question: str,
        reason: str,
        evidence: Sequence[Evidence],
        *,
        open_case: bool = False,
        context: Sequence[ConversationTurn] = (),
        notice: str = "这个问题现有知识不足以保证准确，已转人工负责人确认。",
    ) -> AnswerResult:
        redacted = _redact_question(question)
        handoff = HandoffRequest(
            assignee_user_id=self.operator_user_id,
            reason=reason,
            question_summary=redacted,
            evidence_ids=tuple(item.knowledge_id for item in evidence),
            open_case=open_case,
            context=tuple(context),
        )
        return AnswerResult(
            text=notice,
            evidence=tuple(evidence),
            handoff=handoff,
        )
