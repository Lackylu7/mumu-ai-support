from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from threading import Lock
from typing import Any

from .case_domain import CaseRecord
from .runtime_state import atomic_write_json, read_json_object


class CaseReminderError(RuntimeError):
    pass


class CaseDraftReminder:
    """Create one DingTalk todo per draft lifecycle and close it after publication."""

    def __init__(
        self,
        client: Any,
        *,
        executor_user_ids: Sequence[str],
        state_path: Path,
        knowledge_table_url: str,
    ) -> None:
        self.client = client
        self.executor_user_ids = tuple(
            dict.fromkeys(user_id.strip() for user_id in executor_user_ids if user_id.strip())
        )
        if not self.executor_user_ids:
            raise ValueError("草稿待办至少需要一名执行人")
        self.state_path = state_path
        self.knowledge_table_url = knowledge_table_url.strip()
        self._lock = Lock()

    def ensure_draft(
        self,
        case: CaseRecord,
        knowledge_id: str,
        reasons: Sequence[str],
    ) -> str:
        with self._lock:
            state = self._read_state()
            cases = state["cases"]
            entry = cases.get(case.case_id)
            if not isinstance(entry, dict) or bool(entry.get("completed")):
                task_id = self._create_task(case)
                entry = {
                    "task_id": task_id,
                    "comment_added": False,
                    "completed": False,
                }
                cases[case.case_id] = entry
                self._write_state(state)
            task_id = str(entry.get("task_id", "")).strip()
            if not task_id:
                raise CaseReminderError("草稿待办状态缺少 taskId")
            if not bool(entry.get("comment_added")):
                self.client.add_todo_comment(
                    task_id,
                    self._comment(case, knowledge_id, reasons),
                )
                entry["comment_added"] = True
                self._write_state(state)
            return task_id

    def complete(self, case_id: str) -> None:
        with self._lock:
            state = self._read_state()
            entry = state["cases"].get(case_id)
            if not isinstance(entry, dict) or bool(entry.get("completed")):
                return
            task_id = str(entry.get("task_id", "")).strip()
            if not task_id:
                return
            self.client.complete_todo_task(task_id)
            entry["completed"] = True
            self._write_state(state)

    def _create_task(self, case: CaseRecord) -> str:
        question = " ".join(case.question.split())
        summary = question[:36] + ("…" if len(question) > 36 else "")
        payload = self.client.create_todo_task(
            title=f"[木木知识草稿] {summary}",
            executor_user_ids=self.executor_user_ids,
            priority="30",
        )
        result = payload.get("result")
        task_id = result.get("taskId") if isinstance(result, dict) else None
        if not isinstance(task_id, str) or not task_id.strip():
            raise CaseReminderError("创建草稿待办后未返回 taskId")
        return task_id.strip()

    def _comment(
        self,
        case: CaseRecord,
        knowledge_id: str,
        reasons: Sequence[str],
    ) -> str:
        reason_text = "；".join(dict.fromkeys(reason.strip() for reason in reasons if reason.strip()))
        if not reason_text:
            reason_text = "群聊结论不完整或存在冲突，请补充并确认正确答案。"
        return "\n".join(
            (
                f"案件编号：{case.case_id}",
                f"原问题：{case.question}",
                f"需处理：{reason_text}",
                f"知识编号：{knowledge_id}",
                f"AI表：{self.knowledge_table_url}",
                "处理方式：可在知识缺口填写“负责人答复”并勾选“提交沉淀”，或直接修改知识条目后发布。",
            )
        )

    def _read_state(self) -> dict[str, Any]:
        payload = read_json_object(self.state_path) or {}
        cases = payload.get("cases")
        return {
            "version": 1,
            "cases": dict(cases) if isinstance(cases, dict) else {},
        }

    def _write_state(self, state: dict[str, Any]) -> None:
        atomic_write_json(self.state_path, state)
