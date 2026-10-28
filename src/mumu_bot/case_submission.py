from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime
from threading import Event, Lock, Thread
from typing import Any

from .case_domain import CaseMessage


logger = logging.getLogger(__name__)


class CaseSubmissionWorker:
    def __init__(
        self,
        *,
        repository: Any,
        resolution: Any,
        card_gateway: Any | None = None,
        draft_reminder: Any | None = None,
    ) -> None:
        self.repository = repository
        self.resolution = resolution
        self.card_gateway = card_gateway
        self.draft_reminder = draft_reminder
        self._run_lock = Lock()

    def run_once(self, now: datetime) -> int:
        if not self._run_lock.acquire(blocking=False):
            return 0
        completed = 0
        try:
            for case in self.repository.list_submitted_cases():
                answer = case.responsible_answer.strip()
                if not answer:
                    self.repository.record_submission_error(
                        case.case_id,
                        "已勾选提交沉淀，但负责人答复为空。",
                    )
                    continue
                messages = (
                    CaseMessage(
                        message_id=f"table-question:{case.case_id}",
                        sender_user_id=case.asker_user_id,
                        sender_name=case.asker_name or "提问者",
                        text=case.question,
                        created_at=case.created_at,
                    ),
                    CaseMessage(
                        message_id=f"table-answer:{case.case_id}",
                        sender_user_id=case.owner_user_id,
                        sender_name=case.owner_name or "负责人",
                        text=answer,
                        created_at=now,
                    ),
                )
                try:
                    result = self.resolution.resolve(
                        case.case_id,
                        case.owner_user_id,
                        case.owner_name or "负责人",
                        now,
                        messages=messages,
                        trusted_answer=True,
                    )
                    if result.state != "published":
                        self.repository.record_submission_error(
                            case.case_id,
                            "负责人答复未能发布，请稍后重试。",
                        )
                        continue
                    self._update_card(case.case_id, case.group_id, result)
                    completed += 1
                except Exception as error:
                    logger.warning(
                        "case_submission_failed case_id=%s error_type=%s",
                        case.case_id,
                        type(error).__name__,
                    )
                    self.repository.record_submission_error(
                        case.case_id,
                        f"AI表格沉淀失败：{type(error).__name__}",
                    )
            self._sync_draft_reminders()
            return completed
        finally:
            self._run_lock.release()

    def _sync_draft_reminders(self) -> None:
        if self.draft_reminder is None:
            return
        try:
            draft_cases = self.repository.list_draft_cases()
        except Exception as error:
            logger.warning(
                "case_draft_reminder_scan_failed error_type=%s",
                type(error).__name__,
            )
            return
        for case in draft_cases:
            reason = case.notes.removeprefix("已生成草稿：").strip()
            try:
                self.draft_reminder.ensure_draft(
                    case,
                    case.knowledge_id,
                    (reason,) if reason else (),
                )
            except Exception as error:
                logger.warning(
                    "case_draft_reminder_sync_failed case_id=%s error_type=%s",
                    case.case_id,
                    type(error).__name__,
                )

    def _update_card(self, case_id: str, group_id: str, result: Any) -> None:
        if self.card_gateway is None or not group_id:
            return
        if result.state == "published":
            state = "已发布"
            text = f"负责人已通过知识缺口表提交答案。\n\n知识编号：{result.knowledge_id}"
        else:
            state = "已生成草稿"
            text = f"负责人答复已整理为不可检索草稿。\n\n知识编号：{result.knowledge_id}"
        try:
            self.card_gateway.update(
                case_id,
                text,
                state=state,
                button_enabled=False,
            )
        except Exception as error:
            logger.warning(
                "case_submission_card_update_failed case_id=%s error_type=%s",
                case_id,
                type(error).__name__,
            )


class CaseSubmissionRunner:
    def __init__(
        self,
        worker: CaseSubmissionWorker,
        *,
        interval_seconds: int = 15,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if interval_seconds < 1:
            raise ValueError("轮询间隔必须为正整数")
        self.worker = worker
        self.interval_seconds = interval_seconds
        self.clock = clock or (lambda: datetime.now().astimezone())
        self.stop_event = Event()
        self.thread: Thread | None = None

    def start(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            return
        self.thread = Thread(
            target=self._run,
            name="mumu-case-submission",
            daemon=True,
        )
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=max(1, self.interval_seconds + 1))

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                self.worker.run_once(self.clock())
            except Exception as error:
                logger.warning(
                    "case_submission_poll_failed error_type=%s",
                    type(error).__name__,
                )
            self.stop_event.wait(self.interval_seconds)
