from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib


@dataclass(frozen=True)
class CaseMessage:
    message_id: str
    sender_user_id: str
    sender_name: str
    text: str
    created_at: datetime


@dataclass(frozen=True)
class CaseRecord:
    case_id: str
    source_message_id: str
    asker_user_id: str
    asker_name: str
    question: str
    gap_reason: str
    owner_user_id: str
    owner_name: str
    created_at: datetime
    record_id: str = ""
    business_domain: str = "公司通用"
    group_id: str = ""
    group_name: str = ""
    group_created_at: datetime | None = None
    status: str = "处理中"
    knowledge_id: str = ""
    source_id: str = ""
    close_trigger_user_id: str = ""
    close_trigger_name: str = ""
    closed_at: datetime | None = None
    notes: str = ""
    responsible_answer: str = ""
    submission_requested: bool = False


@dataclass(frozen=True)
class ExtractedKnowledge:
    canonical_question: str
    standard_answer: str
    conditions: tuple[str, ...]
    exceptions: tuple[str, ...]
    steps: tuple[str, ...]
    common_questions: tuple[str, ...]
    business_domain: str
    risk_level: str
    source_message_ids: tuple[str, ...]
    source_authors: tuple[str, ...]
    conflicts: tuple[str, ...]
    missing_information: tuple[str, ...]


@dataclass(frozen=True)
class CaseOpenResult:
    ok: bool
    case_id: str
    group_id: str = ""
    group_name: str = ""
    message: str = ""


@dataclass(frozen=True)
class CaseCommandResult:
    ok: bool
    case_id: str
    command: str
    message: str


@dataclass(frozen=True)
class CaseResolutionResult:
    ok: bool
    case_id: str
    state: str
    message: str
    knowledge_id: str = ""


def case_id_for_message(message_id: str, now: datetime) -> str:
    if not message_id.strip():
        raise ValueError("原消息 ID 不能为空")
    digest = hashlib.sha256(message_id.encode("utf-8")).hexdigest()[:10].upper()
    return f"MM-GAP-{now:%Y%m%d}-{digest}"


def source_id_for_case(case_id: str) -> str:
    return f"MUMU-CHAT-{case_id.rsplit('-', 1)[-1]}"


def knowledge_id_for_case(case_id: str) -> str:
    return f"MUMU-AUTO-{case_id.rsplit('-', 1)[-1]}"


def publish_id_for_case(case_id: str) -> str:
    return f"MUMU-PUB-{case_id.rsplit('-', 1)[-1]}"
