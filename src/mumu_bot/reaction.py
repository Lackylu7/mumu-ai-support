from __future__ import annotations

import json
import time
from pathlib import Path
from threading import Lock
from typing import Callable, Protocol

import requests
from dingtalk_stream.utils import DINGTALK_OPENAPI_ENDPOINT

from mumu_kb.dws import DwsError


THINKING_EMOJI = "思考"
ROBOT_THINKING_EMOTION_NAME = "🤔思考中"
ROBOT_THINKING_EMOTION_TYPE = 2
ROBOT_THINKING_TEXT_EMOTION = {
    "emotionId": "2659900",
    "emotionName": ROBOT_THINKING_EMOTION_NAME,
    "text": ROBOT_THINKING_EMOTION_NAME,
    "backgroundId": "im_bg_1",
}


class ReactionPort(Protocol):
    def add_thinking(self, conversation_id: str, message_id: str) -> None: ...

    def remove_thinking(self, conversation_id: str, message_id: str) -> None: ...


class MessageEmojiClient(Protocol):
    def add_message_emoji(
        self,
        conversation_id: str,
        message_id: str,
        emoji: str,
        *,
        timeout_seconds: int | None = None,
    ) -> dict[str, object]: ...

    def remove_message_emoji(
        self,
        conversation_id: str,
        message_id: str,
        emoji: str,
        *,
        timeout_seconds: int | None = None,
    ) -> dict[str, object]: ...


class NullReactionPort:
    def add_thinking(self, conversation_id: str, message_id: str) -> None:
        del conversation_id, message_id

    def remove_thinking(self, conversation_id: str, message_id: str) -> None:
        del conversation_id, message_id


class RobotThinkingReactionPort:
    """Uses the robot-only reaction endpoint for messages sent to the robot.

    A DWS reaction is performed as the logged-in human account.  That account
    is not a participant in every robot direct chat, so it cannot react to the
    callback message.  The robot endpoint is designed for this exact case and
    makes the indicator visible to the person talking with 木木.
    """

    def __init__(
        self,
        *,
        access_token_provider: Callable[[], str],
        robot_code: str,
        http_post: Callable[..., object] = requests.post,
        timeout: float = 3.0,
    ) -> None:
        if not robot_code.strip():
            raise ValueError("robotCode 不能为空")
        self.access_token_provider = access_token_provider
        self.robot_code = robot_code.strip()
        self.http_post = http_post
        self.timeout = timeout

    def add_thinking(self, conversation_id: str, message_id: str) -> None:
        self._send("reply", conversation_id, message_id)

    def remove_thinking(self, conversation_id: str, message_id: str) -> None:
        self._send("recall", conversation_id, message_id)

    def _send(self, action: str, conversation_id: str, message_id: str) -> None:
        token = self.access_token_provider()
        if not token:
            raise RuntimeError("无法获取钉钉 access token")
        response = self.http_post(
            f"{DINGTALK_OPENAPI_ENDPOINT}/v1.0/robot/emotion/{action}",
            headers={
                "Content-Type": "application/json",
                "x-acs-dingtalk-access-token": token,
            },
            json={
                "emotionType": ROBOT_THINKING_EMOTION_TYPE,
                "emotionName": ROBOT_THINKING_EMOTION_NAME,
                "textEmotion": dict(ROBOT_THINKING_TEXT_EMOTION),
                "openConversationId": conversation_id,
                "openMsgId": message_id,
                "robotCode": self.robot_code,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        try:
            payload = response.json()
        except (AttributeError, ValueError):
            return
        if isinstance(payload, dict) and payload.get("success") is False:
            raise RuntimeError(f"机器人表情{action}未成功")


class DwsThinkingReactionPort:
    def __init__(
        self,
        client: MessageEmojiClient,
        *,
        attempts: int = 2,
        timeout_seconds: int = 2,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if attempts < 1:
            raise ValueError("表情重试次数必须为正数")
        self.client = client
        self.attempts = attempts
        self.timeout_seconds = timeout_seconds
        self.sleeper = sleeper

    def add_thinking(self, conversation_id: str, message_id: str) -> None:
        self._retry(
            lambda: self.client.add_message_emoji(
                conversation_id,
                message_id,
                THINKING_EMOJI,
                timeout_seconds=self.timeout_seconds,
            )
        )

    def remove_thinking(self, conversation_id: str, message_id: str) -> None:
        self._retry(
            lambda: self.client.remove_message_emoji(
                conversation_id,
                message_id,
                THINKING_EMOJI,
                timeout_seconds=self.timeout_seconds,
            )
        )

    def _retry(self, operation: Callable[[], object]) -> None:
        for attempt in range(self.attempts):
            try:
                operation()
                return
            except DwsError:
                if attempt + 1 >= self.attempts:
                    raise
                self.sleeper(0.2 * (attempt + 1))


class TrackedThinkingReactionPort:
    """Tracks active reactions so a restart can remove any stale indicator."""

    def __init__(self, delegate: ReactionPort, state_path: Path) -> None:
        self.delegate = delegate
        self.state_path = state_path
        self._lock = Lock()
        self._active = self._load()

    def cleanup_stale(self) -> None:
        for conversation_id, message_id in tuple(self._active):
            try:
                self.delegate.remove_thinking(conversation_id, message_id)
            except Exception:
                continue
            self._forget(conversation_id, message_id)

    def add_thinking(self, conversation_id: str, message_id: str) -> None:
        self._remember(conversation_id, message_id)
        try:
            self.delegate.add_thinking(conversation_id, message_id)
        except Exception:
            self._forget(conversation_id, message_id)
            raise

    def remove_thinking(self, conversation_id: str, message_id: str) -> None:
        self.delegate.remove_thinking(conversation_id, message_id)
        self._forget(conversation_id, message_id)

    def _load(self) -> set[tuple[str, str]]:
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            return {
                (str(item["conversation_id"]), str(item["message_id"]))
                for item in payload.get("active", [])
            }
        except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError):
            return set()

    def _remember(self, conversation_id: str, message_id: str) -> None:
        with self._lock:
            self._active.add((conversation_id, message_id))
            self._save()

    def _forget(self, conversation_id: str, message_id: str) -> None:
        with self._lock:
            self._active.discard((conversation_id, message_id))
            self._save()

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps(
                {
                    "active": [
                        {
                            "conversation_id": conversation_id,
                            "message_id": message_id,
                        }
                        for conversation_id, message_id in sorted(self._active)
                    ]
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
