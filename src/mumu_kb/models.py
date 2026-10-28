from __future__ import annotations

import json
import re
from dataclasses import MISSING, dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar, get_type_hints


class SourceStatus(str, Enum):
    ORGANIZING = "待整理"
    REVIEW = "待审核"
    PUBLISHED = "已发布"
    DISABLED = "已停用"


class KnowledgeStatus(str, Enum):
    DRAFT = "草稿"
    REVIEW = "待审核"
    PUBLISHED = "已发布"
    DISABLED = "已停用"


class InterpretationBasis(str, Enum):
    EXPLICIT = "文档明确"
    DERIVED = "效果推导"
    PENDING = "待确认"


class InterpretationStatus(str, Enum):
    PUBLISHED = "已发布"
    PENDING = "待确认"
    DISABLED = "已停用"


class RiskLevel(str, Enum):
    NORMAL = "普通"
    IMPORTANT = "重要"
    HIGH = "高风险"


class Sensitivity(str, Enum):
    PUBLIC_INTERNAL = "公开内部"
    INTERNAL = "限内部"


class AnswerBoundary(str, Enum):
    DIRECT = "可直接回答"
    VERIFY_PERSONAL_DATA = "需核验个人数据"
    VERIFY_CURRENT_RULE = "需确认当前规则"
    HUMAN_DECISION = "需负责人判断"


class ReviewIssueStatus(str, Enum):
    PENDING_OPERATOR = "待人工负责人确认"
    RESOLVED = "已解决"


class TestCategory(str, Enum):
    DIRECT_FACT = "直接事实"
    MULTI_CONDITION = "多条件判断"
    EXCEPTION = "例外规则"
    TIME_WINDOW = "时间窗口"
    CALCULATION = "数字计算"
    REGION = "地域差异"
    CROSS_SECTION = "跨章节组合"
    AMBIGUITY = "歧义转人工"
    UNCOVERED = "未覆盖转人工"
    INDUCED_FABRICATION = "诱导编造"
    GENERAL_CHAT = "普通聊天"
    CONTEXT_FOLLOWUP = "连续追问"


class ExpectedAction(str, Enum):
    ANSWER = "回答"
    HANDOFF = "转人工"


class TestReviewStatus(str, Enum):
    REVIEW = "待审核"
    APPROVED = "已审核"


@dataclass(frozen=True)
class SourceRecord:
    source_id: str
    title: str
    business_domain: str
    source_type: str
    source_url: str
    source_node_id: str
    workspace_id: str
    source_owner: str
    source_updated_at: str
    version: str
    status: SourceStatus
    effective_from: str = ""
    effective_to: str = ""
    raw_snapshot: str = ""
    review_notes: str = ""

    def validate(self) -> None:
        if not self.source_id.startswith("EP3-S"):
            raise ValueError("source_id must start with EP3-S")
        if not self.title or not self.source_url:
            raise ValueError("知识来源必须包含标题和链接")


@dataclass(frozen=True)
class KnowledgeEntry:
    knowledge_id: str
    business_domain: str
    intent: str
    canonical_answer: str
    source_id: str
    source_section: str
    status: KnowledgeStatus
    business_topic: str = ""
    question_variants: list[str] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)
    exceptions: list[str] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)
    effective_from: str = ""
    effective_to: str = ""
    risk_level: RiskLevel = RiskLevel.NORMAL
    sensitivity: Sensitivity = Sensitivity.INTERNAL
    answer_boundary: AnswerBoundary = AnswerBoundary.DIRECT
    reviewer: str = ""
    version: str = "1.0"

    def validate(self) -> None:
        if not self.knowledge_id.startswith("EP3-K"):
            raise ValueError("knowledge_id must start with EP3-K")
        if not self.business_domain or not self.intent or not self.source_section:
            raise ValueError("知识条目缺少业务域、意图或来源章节")
        if self.status is KnowledgeStatus.PUBLISHED:
            if not self.canonical_answer or not self.source_id or not self.reviewer:
                raise ValueError("已发布知识必须包含标准答案、来源和审核人")


@dataclass(frozen=True)
class InterpretationEntry:
    interpretation_id: str
    knowledge_ids: list[str]
    business_domain: str
    audience: str
    employee_concern: str
    rationale: str
    employee_value: str
    recommended_response: str
    basis: InterpretationBasis
    source_label: str
    source_section: str
    status: InterpretationStatus
    employee_cost: str = ""
    boundaries: list[str] = field(default_factory=list)
    version: str = "1.0"

    def validate(self) -> None:
        if not self.interpretation_id.startswith(("EP3-I", "MUMU-I-")):
            raise ValueError("interpretation_id must start with EP3-I or MUMU-I-")
        if not self.knowledge_ids or any(
            not knowledge_id.strip()
            or any(character.isspace() for character in knowledge_id)
            or len(knowledge_id) > 100
            for knowledge_id in self.knowledge_ids
        ):
            raise ValueError("制度解读必须关联有效的正式知识编号")
        required = (
            self.business_domain,
            self.audience,
            self.employee_concern,
            self.rationale,
            self.employee_value,
            self.recommended_response,
            self.source_label,
            self.source_section,
        )
        if not all(value.strip() for value in required):
            raise ValueError("制度解读缺少必要内容")
        if self.status is InterpretationStatus.PUBLISHED:
            if self.basis is InterpretationBasis.PENDING:
                raise ValueError("依据待确认的制度解读不得发布")
            if not self.boundaries:
                raise ValueError("已发布制度解读必须包含表达边界")


@dataclass(frozen=True)
class ReviewIssue:
    review_id: str
    source_id: str
    source_section: str
    quoted_text: str
    issue_type: str
    question_for_owner: str
    status: ReviewIssueStatus
    blocked_knowledge_ids: list[str] = field(default_factory=list)

    def validate(self) -> None:
        if not self.review_id.startswith("EP3-R"):
            raise ValueError("review_id must start with EP3-R")
        if not self.question_for_owner:
            raise ValueError("审核问题必须包含向负责人确认的问题")


@dataclass(frozen=True)
class TestCase:
    test_id: str
    category: TestCategory
    question: str
    expected_points: list[str]
    forbidden_content: list[str]
    knowledge_ids: list[str]
    risk_level: RiskLevel
    expected_action: ExpectedAction
    review_status: TestReviewStatus
    context: list[str] = field(default_factory=list)

    def validate(self) -> None:
        if not re.fullmatch(r"[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)+", self.test_id):
            raise ValueError("test_id must be a stable uppercase identifier")
        if not self.question or not self.expected_points:
            raise ValueError("测试题必须包含问题和答案要点")
        if self.risk_level is RiskLevel.HIGH and not self.forbidden_content:
            raise ValueError("高风险测试题必须定义禁止内容")


@dataclass(frozen=True)
class PublishRecord:
    publish_id: str
    business_domain: str
    version: str
    entry_count: int
    published_at: str
    reviewer: str
    summary: str
    knowledge_ids: list[str]
    verification_summary: str

    def validate(self) -> None:
        if not self.publish_id or self.entry_count != len(self.knowledge_ids):
            raise ValueError("发布记录数量与知识编号不一致")


RecordType = TypeVar(
    "RecordType",
    SourceRecord,
    KnowledgeEntry,
    InterpretationEntry,
    ReviewIssue,
    TestCase,
    PublishRecord,
)


def _enum_type(annotation: Any) -> type[Enum] | None:
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return annotation
    return None


def _from_dict(record_type: type[RecordType], value: dict[str, Any]) -> RecordType:
    definitions = {definition.name: definition for definition in fields(record_type)}
    unknown = sorted(set(value) - set(definitions))
    if unknown:
        raise ValueError(f"未知字段: {', '.join(unknown)}")

    missing = [
        name
        for name, definition in definitions.items()
        if definition.default is MISSING
        and definition.default_factory is MISSING
        and name not in value
    ]
    if missing:
        raise ValueError(f"缺少字段: {', '.join(missing)}")

    hints = get_type_hints(record_type)
    converted = dict(value)
    for name, raw_value in value.items():
        enum_type = _enum_type(hints[name])
        if enum_type is not None:
            try:
                converted[name] = enum_type(raw_value)
            except (TypeError, ValueError) as error:
                raise ValueError(f"{name} 包含未知枚举值: {raw_value}") from error

    record = record_type(**converted)
    record.validate()
    return record


def record_from_dict(
    record_type: type[RecordType],
    value: dict[str, Any],
) -> RecordType:
    return _from_dict(record_type, value)


def load_json_records(path: Path, record_type: type[RecordType]) -> list[RecordType]:
    try:
        raw_records = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"无法读取 JSON: {path}") from error
    if not isinstance(raw_records, list):
        raise ValueError(f"JSON 顶层必须是数组: {path}")

    id_field = {
        SourceRecord: "source_id",
        KnowledgeEntry: "knowledge_id",
        InterpretationEntry: "interpretation_id",
        ReviewIssue: "review_id",
        TestCase: "test_id",
        PublishRecord: "publish_id",
    }[record_type]
    records: list[RecordType] = []
    stable_ids: set[str] = set()
    for index, raw_record in enumerate(raw_records, start=1):
        if not isinstance(raw_record, dict):
            raise ValueError(f"第 {index} 条记录必须是对象")
        record = _from_dict(record_type, raw_record)
        stable_id = getattr(record, id_field)
        if stable_id in stable_ids:
            raise ValueError(f"发现重复稳定编号: {stable_id}")
        stable_ids.add(stable_id)
        records.append(record)
    return records
