import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from mumu_kb.dws import DwsError
from mumu_bot.reaction import (
    DwsThinkingReactionPort,
    NullReactionPort,
    RobotThinkingReactionPort,
    TrackedThinkingReactionPort,
)


class FakeDwsClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str, str]] = []

    def add_message_emoji(
        self,
        conversation_id: str,
        message_id: str,
        emoji: str,
        *,
        timeout_seconds: int | None = None,
    ) -> dict[str, object]:
        del timeout_seconds
        self.calls.append(("add", conversation_id, message_id, emoji))
        return {"success": True}

    def remove_message_emoji(
        self,
        conversation_id: str,
        message_id: str,
        emoji: str,
        *,
        timeout_seconds: int | None = None,
    ) -> dict[str, object]:
        del timeout_seconds
        self.calls.append(("remove", conversation_id, message_id, emoji))
        return {"success": True}


class ThinkingReactionPortTests(unittest.TestCase):
    def test_robot_port_uses_robot_scoped_reaction_endpoints(self) -> None:
        calls: list[dict[str, object]] = []

        class Response:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict[str, bool]:
                return {"success": True}

        def post(url: str, **kwargs: object) -> Response:
            calls.append({"url": url, **kwargs})
            return Response()

        port = RobotThinkingReactionPort(
            access_token_provider=lambda: "token",
            robot_code="robot-code",
            http_post=post,
        )

        port.add_thinking("conversation-1", "message-1")
        port.remove_thinking("conversation-1", "message-1")

        self.assertTrue(str(calls[0]["url"]).endswith("/robot/emotion/reply"))
        self.assertTrue(str(calls[1]["url"]).endswith("/robot/emotion/recall"))
        self.assertEqual(
            calls[0]["json"],
            {
                "emotionType": 2,
                "emotionName": "🤔思考中",
                "textEmotion": {
                    "emotionId": "2659900",
                    "emotionName": "🤔思考中",
                    "text": "🤔思考中",
                    "backgroundId": "im_bg_1",
                },
                "openConversationId": "conversation-1",
                "openMsgId": "message-1",
                "robotCode": "robot-code",
            },
        )

    def test_dws_port_adds_and_removes_the_thinking_emoji(self) -> None:
        client = FakeDwsClient()
        port = DwsThinkingReactionPort(client)

        port.add_thinking("conversation-1", "message-1")
        port.remove_thinking("conversation-1", "message-1")

        self.assertEqual(
            client.calls,
            [
                ("add", "conversation-1", "message-1", "思考"),
                ("remove", "conversation-1", "message-1", "思考"),
            ],
        )

    def test_null_port_accepts_the_same_lifecycle_without_side_effects(self) -> None:
        port = NullReactionPort()

        port.add_thinking("conversation-1", "message-1")
        port.remove_thinking("conversation-1", "message-1")

    def test_dws_port_retries_a_transient_reaction_failure(self) -> None:
        class RetryClient(FakeDwsClient):
            def __init__(self) -> None:
                super().__init__()
                self.failures_remaining = 1

            def add_message_emoji(
                self,
                conversation_id: str,
                message_id: str,
                emoji: str,
                *,
                timeout_seconds: int | None = None,
            ) -> dict[str, object]:
                if self.failures_remaining:
                    self.failures_remaining -= 1
                    raise DwsError("temporary failure")
                return super().add_message_emoji(
                    conversation_id,
                    message_id,
                    emoji,
                    timeout_seconds=timeout_seconds,
                )

        sleeps: list[float] = []
        client = RetryClient()
        DwsThinkingReactionPort(client, sleeper=sleeps.append).add_thinking(
            "conversation-1", "message-1"
        )

        self.assertEqual(sleeps, [0.2])
        self.assertEqual(client.calls, [("add", "conversation-1", "message-1", "思考")])

    def test_tracked_port_removes_stale_reaction_after_restart(self) -> None:
        client = FakeDwsClient()
        with TemporaryDirectory() as directory:
            state_path = Path(directory) / "active-thinking.json"
            initial_port = TrackedThinkingReactionPort(
                DwsThinkingReactionPort(client), state_path
            )
            initial_port.add_thinking("conversation-1", "message-1")

            restarted_port = TrackedThinkingReactionPort(
                DwsThinkingReactionPort(client), state_path
            )
            restarted_port.cleanup_stale()

            self.assertEqual(
                client.calls,
                [
                    ("add", "conversation-1", "message-1", "思考"),
                    ("remove", "conversation-1", "message-1", "思考"),
                ],
            )
            self.assertEqual(restarted_port._active, set())


if __name__ == "__main__":
    unittest.main()
