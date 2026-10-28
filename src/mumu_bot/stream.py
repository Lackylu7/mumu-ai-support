from __future__ import annotations

import asyncio
import logging
import re
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from threading import Lock
from typing import Protocol

import requests
from dingtalk_stream import CallbackMessage, ChatbotHandler, ChatbotMessage

from .domain import AnswerResult
from .case_domain import CaseOpenResult
from .reaction import NullReactionPort, ReactionPort


logger = logging.getLogger(__name__)


class SupportServicePort(Protocol):
    def answer(
        self,
        user_id: str,
        question: str,
        now: datetime,
        conversation_id: str = "",
        company_name: str = "",
    ) -> AnswerResult: ...


class ReplyPort(Protocol):
    def reply(
        self,
        incoming: ChatbotMessage,
        text: str,
        at_user_ids: tuple[str, ...],
    ) -> None: ...


class CaseCoordinatorPort(Protocol):
    def handles_group(self, conversation_id: str) -> bool: ...

    def open_case(self, **message_fields: object) -> CaseOpenResult: ...


class HttpResponse(Protocol):
    def raise_for_status(self) -> None: ...


class SessionWebhookReplyPort:
    def __init__(
        self,
        *,
        http_post: Callable[..., HttpResponse] = requests.post,
        timeout: float = 10.0,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.http_post = http_post
        self.timeout = timeout
        self.sleeper = sleeper

    def reply(
        self,
        incoming: ChatbotMessage,
        text: str,
        at_user_ids: tuple[str, ...],
    ) -> None:
        if not incoming.session_webhook:
            raise ValueError("钉钉消息缺少 sessionWebhook")
        payload = {
            "msgtype": "text",
            "text": {"content": text},
            "at": {"atUserIds": list(at_user_ids), "isAtAll": False},
        }
        retry_delays = (0.2, 0.5)
        for attempt in range(len(retry_delays) + 1):
            try:
                response = self.http_post(
                    incoming.session_webhook,
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                return
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as error:
                if attempt >= len(retry_delays):
                    raise
                logger.warning(
                    "session_webhook_transient_failure attempt=%d error_type=%s",
                    attempt + 1,
                    type(error).__name__,
                )
                self.sleeper(retry_delays[attempt])


class MessageDeduplicator:
    def __init__(self, capacity: int = 5000) -> None:
        self.capacity = capacity
        self.message_ids: OrderedDict[str, None] = OrderedDict()

    def add(self, message_id: str) -> bool:
        if message_id in self.message_ids:
            return False
        self.message_ids[message_id] = None
        while len(self.message_ids) > self.capacity:
            self.message_ids.popitem(last=False)
        return True


@dataclass
class PendingRequest:
    key: str
    question: str
    needs_anchor: bool = False
    previous_question: str = ""


class PendingRequestTracker:
    def __init__(self) -> None:
        self._requests: dict[str, list[PendingRequest]] = {}
        self._lock = Lock()

    def register(
        self,
        user_id: str,
        conversation_id: str,
        question: str,
    ) -> PendingRequest:
        key = f"{conversation_id or 'direct'}:{user_id}"
        with self._lock:
            pending = self._requests.setdefault(key, [])
            request = PendingRequest(
                key,
                question,
                needs_anchor=bool(pending),
                previous_question=pending[-1].question if pending else "",
            )
            if pending:
                for existing in pending:
                    existing.needs_anchor = True
            pending.append(request)
            return request

    def complete(self, request: PendingRequest) -> None:
        with self._lock:
            pending = self._requests.get(request.key)
            if pending is None:
                return
            pending[:] = [item for item in pending if item is not request]
            if not pending:
                self._requests.pop(request.key, None)


def _question_anchor(question: str, limit: int = 36) -> str:
    compact = re.sub(r"\s+", " ", question).strip()
    if len(compact) > limit:
        compact = f"{compact[: limit - 1]}…"
    return compact


def _anchor_answer(question: str, answer: str) -> str:
    return f"关于你问的“{_question_anchor(question)}”：\n\n{answer}"


def _append_resolution_prompt(text: str) -> str:
    """Add a compact, explicit escalation affordance to business answers."""
    prompt = "\n\n这条回答解决你的问题了吗？如果没有，请回复“未解决”，木木会建群请负责人协助。"
    if "未解决" in text or "系统暂时繁忙" in text:
        return text
    return f"{text.rstrip()}{prompt}"


def _question_with_pending_context(question: str, previous_question: str) -> str:
    compact = re.sub(r"\s+", "", question)
    if not previous_question or compact.startswith("换一个问题"):
        return question
    if len(compact) > 80 or not re.search(
        r"^(?:那|那么|如果|要是|这样|这种情况|这个|它|还有)|呢[？?]?$",
        compact,
    ):
        return question
    return f"{previous_question}\n用户追问：{question}"


def _company_name_from_message(incoming: ChatbotMessage) -> str:
    """Take a company name only from explicit DingTalk callback fields."""
    values = incoming.extensions
    if not isinstance(values, dict):
        return ""
    for key in (
        "senderCompanyName",
        "senderCorpName",
        "companyName",
        "corpName",
        "orgName",
    ):
        value = values.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


class MumuChatbotHandler(ChatbotHandler):
    def __init__(
        self,
        *,
        service: SupportServicePort,
        reply_port: ReplyPort,
        allowed_user_ids: set[str],
        operator_user_id: str,
        clock: Callable[[], datetime] | None = None,
        deduplicator: MessageDeduplicator | None = None,
        case_coordinator: CaseCoordinatorPort | None = None,
        reaction_port: ReactionPort | None = None,
        pending_requests: PendingRequestTracker | None = None,
        max_concurrent_per_user: int = 3,
    ) -> None:
        super().__init__()
        if max_concurrent_per_user < 1:
            raise ValueError("单用户并发数必须为正数")
        self.service = service
        self.reply_port = reply_port
        self.allowed_user_ids = frozenset(allowed_user_ids)
        self.operator_user_id = operator_user_id
        self.clock = clock or (lambda: datetime.now().astimezone())
        self.deduplicator = deduplicator or MessageDeduplicator()
        self.case_coordinator = case_coordinator
        self.reaction_port = reaction_port or NullReactionPort()
        self.pending_requests = pending_requests or PendingRequestTracker()
        self.max_concurrent_per_user = max_concurrent_per_user
        self._answer_semaphores: dict[str, asyncio.Semaphore] = {}

    async def process(self, callback: CallbackMessage) -> tuple[int, str]:
        incoming = ChatbotMessage.from_dict(callback.data)
        message_id = incoming.message_id or callback.headers.message_id
        if not message_id:
            return 400, "missing message id"
        if not self.deduplicator.add(message_id):
            return 200, "duplicate"

        sender_identifiers = tuple(
            dict.fromkeys(
                value.strip()
                for value in (
                    incoming.sender_staff_id,
                    incoming.sender_id,
                    getattr(incoming.hosting_context, "user_id", None),
                )
                if isinstance(value, str) and value.strip()
            )
        )
        sender_user_id = sender_identifiers[0] if sender_identifiers else ""
        text_parts = incoming.get_text_list() or []
        question = "\n".join(part.strip() for part in text_parts if part.strip())
        conversation_id = incoming.conversation_id or ""
        if (
            self.case_coordinator is not None
            and conversation_id
            and await asyncio.to_thread(
                self.case_coordinator.handles_group,
                conversation_id,
            )
        ):
            return 200, "OK"

        if not set(sender_identifiers).intersection(self.allowed_user_ids):
            logger.warning(
                "access_denied sender_identifiers=%s message_id=%s",
                sender_identifiers,
                message_id,
            )
            await asyncio.to_thread(
                self.reply_port.reply,
                incoming,
                "木木目前仅向首批测试人员开放，暂未开放给当前账号。",
                (sender_user_id,),
            )
            return 200, "OK"

        if not question:
            await asyncio.to_thread(
                self.reply_port.reply,
                incoming,
                "木木目前只处理文字问题，请用文字描述。",
                (sender_user_id,),
            )
            return 200, "OK"

        pending_request = self.pending_requests.register(
            sender_user_id,
            conversation_id,
            question,
        )
        service_question = _question_with_pending_context(
            question,
            pending_request.previous_question,
        )
        company_name = _company_name_from_message(incoming)
        answer_semaphore = self._answer_semaphores.setdefault(
            sender_user_id,
            asyncio.Semaphore(self.max_concurrent_per_user),
        )
        reaction_added = False
        answer_delivered = False
        if conversation_id:
            try:
                await asyncio.to_thread(
                    self.reaction_port.add_thinking,
                    conversation_id,
                    message_id,
                )
                reaction_added = True
            except Exception as error:
                logger.warning(
                    "reaction_add_failed error_type=%s error=%s",
                    type(error).__name__,
                    str(error)[:500],
                )
        try:
            async with answer_semaphore:
                service_args: dict[str, object] = {"conversation_id": conversation_id}
                if company_name:
                    service_args["company_name"] = company_name
                result = await asyncio.to_thread(
                    self.service.answer,
                    sender_user_id,
                    service_question,
                    self.clock(),
                    **service_args,
                )
                text = result.text
                mentions = (sender_user_id,)
                if result.handoff is not None:
                    evidence_ids = "、".join(result.handoff.evidence_ids) or "无"
                    text = (
                        f"{result.text}\n\n"
                        f"@人工负责人 转人工原因：{result.handoff.reason}\n"
                        f"问题摘要：{result.handoff.question_summary}\n"
                        f"已命中知识：{evidence_ids}"
                    )
                    mentions = (sender_user_id, self.operator_user_id)
                if pending_request.needs_anchor:
                    text = _anchor_answer(question, text)
                if result.evidence and result.handoff is None:
                    text = _append_resolution_prompt(text)
                await asyncio.to_thread(
                    self.reply_port.reply,
                    incoming,
                    text,
                    mentions,
                )
                answer_delivered = True
                if (
                    self.case_coordinator is not None
                    and result.handoff is not None
                    and result.handoff.open_case
                ):
                    case_result = await asyncio.to_thread(
                        self.case_coordinator.open_case,
                        message_id=message_id,
                        conversation_id=conversation_id,
                        asker_user_id=sender_user_id,
                        asker_name=incoming.sender_nick or sender_user_id,
                        question=question,
                        handoff=result.handoff,
                        now=self.clock(),
                    )
                    await asyncio.to_thread(
                        self.reply_port.reply,
                        incoming,
                        (
                            _anchor_answer(question, case_result.message)
                            if pending_request.needs_anchor
                            else case_result.message
                        ),
                        (sender_user_id,),
                    )
        except Exception as error:
            logger.warning(
                "answer_processing_failed error_type=%s error=%s",
                type(error).__name__,
                str(error).replace("\n", " ")[:300],
            )
            if not answer_delivered:
                try:
                    await asyncio.to_thread(
                        self.reply_port.reply,
                        incoming,
                        "系统暂时繁忙，请稍后重试。",
                        (sender_user_id,),
                    )
                except Exception as reply_error:
                    logger.warning(
                        "busy_reply_failed error_type=%s",
                        type(reply_error).__name__,
                    )
        finally:
            if reaction_added:
                try:
                    await asyncio.to_thread(
                        self.reaction_port.remove_thinking,
                        conversation_id,
                        message_id,
                    )
                except Exception as error:
                    logger.warning(
                        "reaction_remove_failed error_type=%s",
                        type(error).__name__,
                    )
            self.pending_requests.complete(pending_request)
        return 200, "OK"
