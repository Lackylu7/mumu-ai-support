import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from mumu_bot.domain import ConversationTurn, QuestionScope
from mumu_bot.memory import ConversationMemory


def turn(index: int, created_at: datetime) -> ConversationTurn:
    return ConversationTurn(
        question=f"问题{index}",
        answer=f"答案{index}",
        scope=QuestionScope.GENERAL_CHAT,
        created_at=created_at,
    )


class ConversationMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.now = datetime(2026, 8, 10, 6, 0, tzinfo=timezone.utc)

    def test_keeps_last_ten_turns_and_isolates_session_keys(self) -> None:
        memory = ConversationMemory(max_turns=10)
        for index in range(12):
            memory.append("conversation-a:user-1", turn(index, self.now))
        memory.append("conversation-b:user-1", turn(20, self.now))
        memory.append("conversation-a:user-2", turn(30, self.now))

        history = memory.history("conversation-a:user-1", self.now)

        self.assertEqual([item.question for item in history], [f"问题{i}" for i in range(2, 12)])
        self.assertEqual(
            [item.question for item in memory.history("conversation-b:user-1", self.now)],
            ["问题20"],
        )
        self.assertEqual(
            [item.question for item in memory.history("conversation-a:user-2", self.now)],
            ["问题30"],
        )

    def test_history_uses_three_turn_fallback_after_thirty_minutes(self) -> None:
        memory = ConversationMemory(ttl=timedelta(minutes=30))
        for index in range(5):
            memory.append("conversation-a:user-1", turn(index, self.now))

        active = memory.history(
            "conversation-a:user-1",
            self.now + timedelta(minutes=30),
        )
        expired = memory.history(
            "conversation-a:user-1",
            self.now + timedelta(minutes=30, seconds=1),
        )

        self.assertEqual([item.question for item in active], ["问题0", "问题1", "问题2", "问题3", "问题4"])
        self.assertEqual(
            [item.question for item in expired],
            ["问题2", "问题3", "问题4"],
        )

    def test_stale_fallback_turns_can_be_configured(self) -> None:
        memory = ConversationMemory(
            max_turns=10,
            ttl=timedelta(minutes=30),
            stale_fallback_turns=2,
        )
        for index in range(4):
            memory.append("conversation-a:user-1", turn(index, self.now))

        history = memory.history(
            "conversation-a:user-1",
            self.now + timedelta(hours=1),
        )

        self.assertEqual([item.question for item in history], ["问题2", "问题3"])

    def test_recent_three_turns_survive_process_restart(self) -> None:
        with TemporaryDirectory() as directory:
            state_path = Path(directory) / "conversation-memory.json"
            memory = ConversationMemory(state_path=state_path)
            for index in range(5):
                memory.append("conversation-a:user-1", turn(index, self.now))

            restored = ConversationMemory(state_path=state_path)
            history = restored.history(
                "conversation-a:user-1",
                self.now + timedelta(hours=1),
            )

        self.assertEqual(
            [item.question for item in history],
            ["问题2", "问题3", "问题4"],
        )

    def test_empty_or_unknown_session_has_no_history(self) -> None:
        memory = ConversationMemory()

        self.assertEqual(memory.history("", self.now), ())
        self.assertEqual(memory.history("missing", self.now), ())

    def test_parallel_answers_are_remembered_by_question_time_not_completion_order(self) -> None:
        memory = ConversationMemory()
        memory.append(
            "conversation-a:user-1",
            turn(2, self.now + timedelta(seconds=2)),
        )
        memory.append("conversation-a:user-1", turn(1, self.now))

        history = memory.history(
            "conversation-a:user-1",
            self.now + timedelta(seconds=3),
        )

        self.assertEqual([item.question for item in history], ["问题1", "问题2"])


if __name__ == "__main__":
    unittest.main()
