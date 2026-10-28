from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta
import json
from pathlib import Path
from threading import Lock

from .domain import ConversationTurn, QuestionScope


class ConversationMemory:
    def __init__(
        self,
        max_turns: int = 10,
        ttl: timedelta = timedelta(minutes=30),
        stale_fallback_turns: int = 3,
        state_path: Path | None = None,
    ) -> None:
        if max_turns < 1 or ttl <= timedelta(0) or stale_fallback_turns < 1:
            raise ValueError("会话容量和有效期必须为正数")
        if stale_fallback_turns > max_turns:
            raise ValueError("超时回看轮数不能超过会话容量")
        self.max_turns = max_turns
        self.ttl = ttl
        self.stale_fallback_turns = stale_fallback_turns
        self.state_path = state_path
        self._turns: dict[str, deque[ConversationTurn]] = {}
        self._last_activity: dict[str, datetime] = {}
        self._lock = Lock()
        self._load()

    def history(
        self,
        session_key: str,
        now: datetime,
    ) -> tuple[ConversationTurn, ...]:
        if not session_key:
            return ()
        with self._lock:
            last_activity = self._last_activity.get(session_key)
            if last_activity is None:
                return ()
            turns = tuple(self._turns.get(session_key, ()))
            if now - last_activity > self.ttl:
                # 超过 30 分钟后仍保留少量上下文，专门用于识别“还有什么呢”
                # 这类追问；不把完整旧会话重新塞回模型，避免影响速度和主题判断。
                return turns[-self.stale_fallback_turns :]
            return turns

    def append(self, session_key: str, turn: ConversationTurn) -> None:
        if not session_key:
            return
        with self._lock:
            turns = self._turns.get(session_key)
            if turns is None:
                turns = deque(maxlen=self.max_turns)
            ordered = sorted((*turns, turn), key=lambda item: item.created_at)
            self._turns[session_key] = deque(
                ordered[-self.max_turns :],
                maxlen=self.max_turns,
            )
            self._last_activity[session_key] = max(
                self._last_activity.get(session_key, turn.created_at),
                turn.created_at,
            )
            self._save()

    def _load(self) -> None:
        if self.state_path is None or not self.state_path.exists():
            return
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        sessions = payload.get("sessions") if isinstance(payload, dict) else None
        if not isinstance(sessions, dict):
            return
        for session_key, raw_turns in sessions.items():
            if not isinstance(session_key, str) or not isinstance(raw_turns, list):
                continue
            parsed: list[ConversationTurn] = []
            for raw in raw_turns[-self.stale_fallback_turns :]:
                if not isinstance(raw, dict):
                    continue
                try:
                    parsed.append(
                        ConversationTurn(
                            question=str(raw["question"]),
                            answer=str(raw["answer"]),
                            scope=QuestionScope(str(raw["scope"])),
                            created_at=datetime.fromisoformat(str(raw["created_at"])),
                        )
                    )
                except (KeyError, TypeError, ValueError):
                    continue
            if not parsed:
                continue
            parsed.sort(key=lambda item: item.created_at)
            self._turns[session_key] = deque(parsed, maxlen=self.max_turns)
            self._last_activity[session_key] = parsed[-1].created_at

    def _save(self) -> None:
        if self.state_path is None:
            return
        sessions = {
            session_key: [
                {
                    "question": turn.question,
                    "answer": turn.answer,
                    "scope": turn.scope.value,
                    "created_at": turn.created_at.isoformat(),
                }
                for turn in tuple(turns)[-self.stale_fallback_turns :]
            ]
            for session_key, turns in self._turns.items()
            if turns
        }
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
            temporary.write_text(
                json.dumps({"sessions": sessions}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temporary.replace(self.state_path)
        except OSError:
            return
