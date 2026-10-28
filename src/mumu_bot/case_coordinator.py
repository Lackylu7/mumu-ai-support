from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any

from .case_domain import (
    CaseCommandResult,
    CaseOpenResult,
    CaseRecord,
    case_id_for_message,
    knowledge_id_for_case,
)
from .deepseek import validate_extracted_knowledge
from .domain import HandoffRequest


logger = logging.getLogger(__name__)


def _safe_topic(question: str) -> str:
    compact = re.sub(r"\s+", "", question)
    compact = re.sub(r"[^\w\u4e00-\u9fff-]", "", compact)
    return compact[:18] or "业务问题"


class CaseCoordinator:
    def __init__(
        self,
        *,
        repository: Any,
        gateway: Any,
        model: Any,
        account_owner_user_id: str,
        account_owner_name: str,
        responsible_user_id: str,
        responsible_name: str,
        max_messages: int,
        knowledge_repository: Any | None = None,
    ) -> None:
        self.repository = repository
        self.gateway = gateway
        self.model = model
        self.account_owner_user_id = account_owner_user_id
        self.account_owner_name = account_owner_name
        self.responsible_user_id = responsible_user_id
        self.responsible_name = responsible_name
        self.max_messages = max_messages
        self.knowledge_repository = knowledge_repository
        self._handled_commands: dict[str, CaseCommandResult] = {}

    def handles_group(self, conversation_id: str) -> bool:
        return self.repository.find_by_group_id(conversation_id) is not None

    @staticmethod
    def _command(text: str) -> str:
        compact = re.sub(r"\s+", "", text)
        compact = compact.replace("@木木", "")
        if compact == "问题已解决":
            return "close"
        if compact == "结论有误":
            return "rollback"
        return ""

    def handle_command(
        self,
        *,
        message_id: str,
        conversation_id: str,
        sender_user_id: str,
        sender_name: str,
        text: str,
        now: datetime,
    ) -> CaseCommandResult | None:
        command = self._command(text)
        if not command:
            return None
        case = self.repository.find_by_group_id(conversation_id)
        if case is None:
            return None
        existing = self._handled_commands.get(message_id)
        if existing is not None:
            return existing
        if command == "rollback":
            result = self._rollback(
                case,
                sender_user_id=sender_user_id,
                sender_name=sender_name,
                now=now,
            )
        else:
            result = self._close(
                case,
                sender_user_id=sender_user_id,
                sender_name=sender_name,
                now=now,
            )
        self._handled_commands[message_id] = result
        return result

    def _close(
        self,
        case: CaseRecord,
        *,
        sender_user_id: str,
        sender_name: str,
        now: datetime,
    ) -> CaseCommandResult:
        if case.status == "已补充" and self.repository.publication_is_valid(case.case_id):
            return CaseCommandResult(True, case.case_id, "close", "该案件已经完成。")
        try:
            raw_messages = self.gateway.list_messages(
                case.group_id,
                case.group_created_at or case.created_at,
                limit=self.max_messages,
            )
            messages = tuple(
                item for item in raw_messages if not self._command(item.text)
            )
            candidate = self.model.extract_case_knowledge(messages)
        except Exception as error:
            logger.warning(
                "case_close_failed case_id=%s step=extract error_type=%s",
                case.case_id,
                type(error).__name__,
            )
            message = "暂时无法读取或整理群聊结论，请稍后重新发送结案口令。"
            self._announce(case.group_id, "木木结案失败", message)
            return CaseCommandResult(False, case.case_id, "close", message)

        validation_errors = list(
            validate_extracted_knowledge(
                candidate,
                messages,
                asker_user_id=case.asker_user_id,
            )
        )
        validation_errors.extend(candidate.conflicts)
        validation_errors.extend(candidate.missing_information)
        validation_errors = list(dict.fromkeys(validation_errors))
        if validation_errors:
            message = "目前还不能自动发布，请继续确认：\n" + "\n".join(
                f"- {item}" for item in validation_errors
            )
            self._announce(case.group_id, "木木需要补充信息", message)
            return CaseCommandResult(False, case.case_id, "close", message)

        try:
            self.repository.ensure_source(case.case_id, candidate, messages, now)
            knowledge_id = self.repository.ensure_knowledge(
                case.case_id,
                candidate,
                trigger_name=sender_name,
                now=now,
            )
        except Exception as error:
            logger.warning(
                "case_close_failed case_id=%s step=publish error_type=%s",
                case.case_id,
                type(error).__name__,
            )
            message = "来源或知识写入失败，已保留可重试状态，请稍后重新结案。"
            self._announce(case.group_id, "木木发布失败", message)
            return CaseCommandResult(False, case.case_id, "close", message)

        if not self.repository.publication_is_valid(case.case_id):
            self.repository.disable_case_knowledge(
                case.case_id,
                trigger_user_id=sender_user_id,
                trigger_name=sender_name,
                now=now,
            )
            self._invalidate_knowledge()
            message = "知识回读校验失败，已立即停用，本案件继续处理中。"
            self._announce(case.group_id, "木木回读校验失败", message)
            return CaseCommandResult(False, case.case_id, "close", message)

        try:
            publish_id = self.repository.ensure_publish_record(
                case.case_id,
                candidate,
                trigger_name=sender_name,
                now=now,
            )
        except Exception as error:
            logger.warning(
                "case_close_failed case_id=%s step=publish_record error_type=%s",
                case.case_id,
                type(error).__name__,
            )
            message = "知识已生成，但发布记录写入失败；案件已保留为可重试状态，请重新发送结案口令。"
            self._announce(case.group_id, "木木发布记录失败", message)
            return CaseCommandResult(False, case.case_id, "close", message)

        self.repository.mark_completed(
            case.case_id,
            trigger_user_id=sender_user_id,
            trigger_name=sender_name,
            now=now,
        )
        self._invalidate_knowledge()
        message = "\n".join(
            (
                "本案件已形成并自动发布知识。",
                f"规范问题：{candidate.canonical_question}",
                f"标准答案：{candidate.standard_answer}",
                f"知识编号：{knowledge_id}",
                f"发布记录：{publish_id}",
                "如结论不正确，请发送：@木木 结论有误",
            )
        )
        self._announce(case.group_id, "木木案件已结案", message)
        return CaseCommandResult(True, case.case_id, "close", message)

    def _rollback(
        self,
        case: CaseRecord,
        *,
        sender_user_id: str,
        sender_name: str,
        now: datetime,
    ) -> CaseCommandResult:
        if case.status != "已补充":
            message = "当前案件没有生效中的自动知识，仍处于处理中。"
            self._announce(case.group_id, "木木案件状态", message)
            return CaseCommandResult(True, case.case_id, "rollback", message)
        self.repository.disable_case_knowledge(
            case.case_id,
            trigger_user_id=sender_user_id,
            trigger_name=sender_name,
            now=now,
        )
        self._invalidate_knowledge()
        message = (
            f"知识 {knowledge_id_for_case(case.case_id)} 已停用，案件已重新打开。"
            "请继续讨论，形成正确结论后再次发送：@木木 问题已解决"
        )
        self._announce(case.group_id, "木木已回滚结论", message)
        return CaseCommandResult(True, case.case_id, "rollback", message)

    def _invalidate_knowledge(self) -> None:
        if self.knowledge_repository is not None:
            self.knowledge_repository.invalidate_snapshot()

    def _announce(self, group_id: str, title: str, message: str) -> None:
        try:
            self.gateway.send_bot_message(group_id, title, message, ())
        except Exception as error:
            logger.warning(
                "case_announce_failed group_id_present=%s error_type=%s",
                bool(group_id),
                type(error).__name__,
            )

    def open_case(
        self,
        *,
        message_id: str,
        conversation_id: str,
        asker_user_id: str,
        asker_name: str,
        question: str,
        handoff: HandoffRequest,
        now: datetime,
    ) -> CaseOpenResult:
        del conversation_id
        case_id = case_id_for_message(message_id, now)
        if not handoff.open_case:
            return CaseOpenResult(False, case_id, message="该人工转接不属于业务知识缺口")
        try:
            case = self.repository.ensure_case(
                CaseRecord(
                    case_id=case_id,
                    source_message_id=message_id,
                    asker_user_id=asker_user_id,
                    asker_name=asker_name,
                    question=handoff.question_summary or question,
                    gap_reason=handoff.reason,
                    owner_user_id=self.responsible_user_id,
                    owner_name=self.responsible_name,
                    created_at=now,
                )
            )
        except Exception as error:
            logger.warning(
                "case_open_failed case_id=%s step=ensure_case error_type=%s",
                case_id,
                type(error).__name__,
            )
            return CaseOpenResult(False, case_id, message="案件记录创建失败，已保留人工转接。")

        group_name = case.group_name or (
            f"木木答疑-{case_id.rsplit('-', 1)[-1][:6]}-{_safe_topic(case.question)}"
        )[:60]
        group_id = case.group_id
        if not group_id:
            try:
                group_id = self.gateway.find_group_by_exact_name(group_name)
                if not group_id:
                    members = tuple(
                        dict.fromkeys(
                            user_id
                            for user_id in (
                                asker_user_id,
                                self.responsible_user_id,
                                self.account_owner_user_id,
                            )
                            if user_id
                        )
                    )
                    group_id = self.gateway.create_group(group_name, members)
                case = self.repository.update_case_group(
                    case_id,
                    group_id=group_id,
                    group_name=group_name,
                    created_at=now,
                )
            except Exception as error:
                logger.warning(
                    "case_open_failed case_id=%s step=create_group error_type=%s",
                    case_id,
                    type(error).__name__,
                )
                self._record_error(case_id, "create_group", error)
                return CaseOpenResult(
                    False,
                    case_id,
                    group_id=group_id,
                    group_name=group_name,
                    message="自动建群失败，已通知人工负责人处理。",
                )

        if case.notes in {"intro_sent", "案件群已创建，问题简介已发送。"}:
            return CaseOpenResult(
                True,
                case_id,
                group_id=group_id,
                group_name=group_name,
                message="案件群已存在。",
            )

        try:
            self.gateway.add_bot(group_id)
        except Exception as error:
            logger.warning(
                "case_open_failed case_id=%s step=add_bot error_type=%s",
                case_id,
                type(error).__name__,
            )
            self._record_error(case_id, "add_bot", error)
            return CaseOpenResult(
                False,
                case_id,
                group_id=group_id,
                group_name=group_name,
                message="群已创建，但木木加入失败；请手动添加木木后重试。",
            )

        brief = self.model.build_case_brief(
            case_id=case_id,
            question=case.question,
            asker_name=asker_name,
            gap_reason=handoff.reason,
            checked_evidence_ids=handoff.evidence_ids,
            context=handoff.context,
        )
        mentions = tuple(
            dict.fromkeys(
                user_id
                for user_id in (
                    asker_user_id,
                    self.responsible_user_id,
                    self.account_owner_user_id,
                )
                if user_id
            )
        )
        last_error: Exception | None = None
        for _ in range(2):
            try:
                self.gateway.send_case_card(
                    group_id,
                    case_id,
                    brief,
                    mentions,
                )
                self.repository.mark_intro_sent(case_id)
                return CaseOpenResult(
                    True,
                    case_id,
                    group_id=group_id,
                    group_name=group_name,
                    message="已创建业务答疑群。",
                )
            except Exception as error:
                last_error = error
        logger.warning(
            "case_open_failed case_id=%s step=send_intro error_type=%s",
            case_id,
            type(last_error).__name__ if last_error is not None else "Unknown",
        )
        if last_error is not None:
            self._record_error(case_id, "send_intro", last_error)
        return CaseOpenResult(
            False,
            case_id,
            group_id=group_id,
            group_name=group_name,
            message="群已创建，但案件简介发送失败；可在重试后继续。",
        )

    def _record_error(self, case_id: str, step: str, error: Exception) -> None:
        human_notes = {
            "create_group": "自动建群失败，请检查建群权限后重试。",
            "add_bot": "机器人加入案件群失败，请检查机器人权限后重试。",
            "send_intro": "案件群已创建，但问题简介发送失败，请稍后重试。",
        }
        try:
            self.repository.update_case_note(
                case_id,
                human_notes.get(step, "案件处理暂时失败，请稍后重试。"),
            )
        except Exception as note_error:
            logger.warning(
                "case_note_failed case_id=%s step=%s error_type=%s",
                case_id,
                step,
                type(note_error).__name__,
            )
