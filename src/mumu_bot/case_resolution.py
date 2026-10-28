from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime
from typing import Any

from .case_domain import CaseMessage, CaseResolutionResult, ExtractedKnowledge
from .deepseek import CASE_BUSINESS_DOMAINS, validate_extracted_knowledge


logger = logging.getLogger(__name__)


class CaseResolutionError(RuntimeError):
    pass


class CaseResolutionService:
    def __init__(
        self,
        *,
        repository: Any,
        gateway: Any,
        model: Any,
        max_messages: int,
        knowledge_repository: Any | None = None,
        draft_reminder: Any | None = None,
    ) -> None:
        self.repository = repository
        self.gateway = gateway
        self.model = model
        self.max_messages = max_messages
        self.knowledge_repository = knowledge_repository
        self.draft_reminder = draft_reminder

    def resolve(
        self,
        case_id: str,
        trigger_user_id: str,
        trigger_name: str,
        now: datetime,
        *,
        messages: Sequence[CaseMessage] | None = None,
        trusted_answer: bool = False,
    ) -> CaseResolutionResult:
        case = self.repository.find_by_case_id(case_id)
        if case is None:
            raise CaseResolutionError("案件不存在")
        if case.status == "已补充" and self.repository.publication_is_valid(case_id):
            self._complete_draft_reminder(case_id)
            return CaseResolutionResult(
                True,
                case_id,
                "published",
                "该案件已经完成知识发布。",
                case.knowledge_id,
            )

        case_messages = tuple(messages) if messages is not None else tuple(
            self.gateway.list_messages(
                case.group_id,
                case.group_created_at or case.created_at,
                limit=self.max_messages,
            )
        )
        if not case_messages:
            raise CaseResolutionError("案件没有可用于沉淀的人类消息")
        case_messages = self._with_case_question_context(case, case_messages)

        if trusted_answer:
            fallback_candidate = self._trusted_answer_candidate(
                case,
                case_messages,
                trigger_name,
            )
            candidate = self._structure_trusted_answer(
                case,
                case_messages,
                fallback_candidate,
            )
            validation_errors: list[str] = []
        else:
            candidate = self.model.extract_case_knowledge(case_messages)
            validation_errors = list(
                validate_extracted_knowledge(
                    candidate,
                    case_messages,
                    asker_user_id=case.asker_user_id,
                )
            )
            validation_errors.extend(candidate.conflicts)
            validation_errors.extend(candidate.missing_information)
        reasons = tuple(dict.fromkeys(item for item in validation_errors if item))
        is_draft = bool(reasons)
        source_status = "待整理" if is_draft else "已发布"
        knowledge_status = "草稿" if is_draft else "已发布"

        self.repository.ensure_source(
            case_id,
            candidate,
            case_messages,
            now,
            status=source_status,
        )
        knowledge_id = self.repository.ensure_knowledge(
            case_id,
            candidate,
            trigger_name=trigger_name,
            now=now,
            status=knowledge_status,
        )

        if is_draft:
            self.repository.mark_draft(
                case_id,
                trigger_user_id=trigger_user_id,
                trigger_name=trigger_name,
                now=now,
                reasons=reasons,
            )
            self._ensure_draft_reminder(case, knowledge_id, reasons)
            return CaseResolutionResult(
                True,
                case_id,
                "draft",
                "群聊信息仍不完整或存在冲突，已生成不可检索的知识草稿。",
                knowledge_id,
            )

        if not self.repository.publication_is_valid(case_id):
            self.repository.disable_case_knowledge(
                case_id,
                trigger_user_id=trigger_user_id,
                trigger_name=trigger_name,
                now=now,
            )
            self._invalidate_knowledge()
            raise CaseResolutionError("知识回读校验失败，已停用本次知识")

        publish_id = self.repository.ensure_publish_record(
            case_id,
            candidate,
            trigger_name=trigger_name,
            now=now,
        )
        self.repository.mark_completed(
            case_id,
            trigger_user_id=trigger_user_id,
            trigger_name=trigger_name,
            now=now,
        )
        self._complete_draft_reminder(case_id)
        self._invalidate_knowledge()
        return CaseResolutionResult(
            True,
            case_id,
            "published",
            f"知识已发布，发布记录：{publish_id}",
            knowledge_id,
        )

    @staticmethod
    def _with_case_question_context(
        case: Any,
        messages: tuple[CaseMessage, ...],
    ) -> tuple[CaseMessage, ...]:
        question = case.question.strip()
        if not question:
            return messages
        if any(item.text.strip() == question for item in messages):
            return messages
        context = CaseMessage(
            message_id=f"case-question:{case.case_id}",
            sender_user_id=case.asker_user_id,
            sender_name=case.asker_name or "提问者",
            text=question,
            created_at=case.created_at,
        )
        return (context, *messages)

    @staticmethod
    def _trusted_answer_candidate(
        case: Any,
        messages: tuple[CaseMessage, ...],
        trigger_name: str,
    ) -> ExtractedKnowledge:
        answer_message = messages[-1]
        answer = answer_message.text.strip()
        if not answer:
            raise CaseResolutionError("负责人答复为空")
        # 提交沉淀时保留案件已判定的业务域；不能把新增业务一律降到通用域。
        business_domain = case.business_domain.strip() or "公司通用"
        author = answer_message.sender_name.strip() or trigger_name or "负责人"
        question = case.question.strip()
        return ExtractedKnowledge(
            canonical_question=question,
            standard_answer=answer,
            conditions=(),
            exceptions=(),
            steps=(),
            common_questions=(question,),
            business_domain=business_domain,
            risk_level="普通",
            source_message_ids=(answer_message.message_id,),
            source_authors=(author,),
            conflicts=(),
            missing_information=(),
        )

    def _structure_trusted_answer(
        self,
        case: Any,
        messages: tuple[CaseMessage, ...],
        fallback: ExtractedKnowledge,
    ) -> ExtractedKnowledge:
        """Structure a responsible answer without ever blocking its publication."""
        try:
            extracted = self.model.extract_case_knowledge(messages)
            business_domain = fallback.business_domain
            if (
                business_domain == "公司通用"
                and extracted.business_domain in CASE_BUSINESS_DOMAINS
            ):
                business_domain = extracted.business_domain
            candidate = replace(
                extracted,
                business_domain=business_domain,
                source_message_ids=fallback.source_message_ids,
                source_authors=fallback.source_authors,
                conflicts=(),
                missing_information=(),
            )
            errors = validate_extracted_knowledge(
                candidate,
                messages,
                asker_user_id=case.asker_user_id,
            )
            if errors:
                logger.warning(
                    "trusted_answer_structuring_rejected case_id=%s reasons=%s",
                    case.case_id,
                    " | ".join(errors),
                )
                return fallback
            return candidate
        except Exception as error:
            logger.warning(
                "trusted_answer_structuring_failed case_id=%s error_type=%s",
                case.case_id,
                type(error).__name__,
            )
            return fallback

    def _invalidate_knowledge(self) -> None:
        if self.knowledge_repository is not None:
            self.knowledge_repository.invalidate_snapshot()

    def _ensure_draft_reminder(
        self,
        case: Any,
        knowledge_id: str,
        reasons: tuple[str, ...],
    ) -> None:
        if self.draft_reminder is None:
            return
        try:
            self.draft_reminder.ensure_draft(case, knowledge_id, reasons)
        except Exception as error:
            logger.warning(
                "case_draft_reminder_failed case_id=%s error_type=%s",
                case.case_id,
                type(error).__name__,
            )

    def _complete_draft_reminder(self, case_id: str) -> None:
        if self.draft_reminder is None:
            return
        try:
            self.draft_reminder.complete(case_id)
        except Exception as error:
            logger.warning(
                "case_draft_reminder_completion_failed case_id=%s error_type=%s",
                case_id,
                type(error).__name__,
            )
