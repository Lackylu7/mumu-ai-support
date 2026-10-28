from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class QuestionScope(str, Enum):
    BUSINESS_MATCHED = "business_matched"
    BUSINESS_GAP = "business_gap"
    GENERAL_CHAT = "general_chat"
    TEMPORARY_ERROR = "temporary_error"


@dataclass(frozen=True)
class InterpretationEvidence:
    interpretation_id: str
    employee_concern: str
    rationale: str
    employee_value: str
    employee_cost: str
    recommended_response: str
    boundaries: tuple[str, ...]
    basis: str
    audience: str
    source_label: str
    source_section: str
    version: str


@dataclass(frozen=True)
class Evidence:
    knowledge_id: str
    intent: str
    answer: str
    source_id: str
    source_section: str
    version: str
    risk_level: str
    sensitivity: str
    conditions: tuple[str, ...] = ()
    exceptions: tuple[str, ...] = ()
    steps: tuple[str, ...] = ()
    common_questions: tuple[str, ...] = ()
    interpretations: tuple[InterpretationEvidence, ...] = ()
    business_domain: str = "公司通用"
    business_topic: str = "未分类"
    answer_boundary: str = "可直接回答"
    # Optional deep link to the original DingTalk document/table source.
    source_url: str = ""


@dataclass(frozen=True)
class RetrievalResult:
    scope: QuestionScope
    evidence: tuple[Evidence, ...] = ()
    reason: str = ""


@dataclass(frozen=True)
class ConversationTurn:
    question: str
    answer: str
    scope: QuestionScope
    created_at: datetime


@dataclass(frozen=True)
class ContextDecision:
    use_history: bool
    standalone_question: str
    interaction_intent: str = "question"


@dataclass(frozen=True)
class HandoffRequest:
    assignee_user_id: str
    reason: str
    question_summary: str
    evidence_ids: tuple[str, ...] = ()
    open_case: bool = False
    context: tuple[ConversationTurn, ...] = ()


@dataclass(frozen=True)
class AnswerResult:
    text: str
    evidence: tuple[Evidence, ...] = ()
    handoff: HandoffRequest | None = None

    @property
    def is_handoff(self) -> bool:
        return self.handoff is not None
