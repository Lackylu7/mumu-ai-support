import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from dingtalk_stream import Credential, DingTalkStreamClient

from mumu_bot.monitored_stream import MonitoredDingTalkStreamClient


class FakeReporter:
    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []

    def connecting(self) -> None:
        self.events.append(("connecting", ""))

    def connected(self) -> None:
        self.events.append(("connected", ""))

    def disconnected(self, error_type: str = "") -> None:
        self.events.append(("disconnected", error_type))


class MonitoredStreamTests(unittest.TestCase):
    def test_successful_open_connection_marks_connected_without_persisting_payload(self) -> None:
        reporter = FakeReporter()
        client = MonitoredDingTalkStreamClient(Credential("id", "secret"), reporter)
        endpoint = {"endpoint": "wss://sensitive.invalid", "ticket": "sensitive-ticket"}

        with patch.object(DingTalkStreamClient, "open_connection", return_value=endpoint):
            result = client.open_connection()

        self.assertIs(result, endpoint)
        self.assertEqual(reporter.events, [("connected", "")])

    def test_failed_open_connection_marks_disconnected(self) -> None:
        reporter = FakeReporter()
        client = MonitoredDingTalkStreamClient(Credential("id", "secret"), reporter)

        with patch.object(DingTalkStreamClient, "open_connection", return_value=None):
            self.assertIsNone(client.open_connection())

        self.assertEqual(
            reporter.events,
            [("disconnected", "OpenConnectionFailed")],
        )

    def test_open_connection_exception_records_only_exception_type_and_reraises(self) -> None:
        reporter = FakeReporter()
        client = MonitoredDingTalkStreamClient(Credential("id", "secret"), reporter)

        with patch.object(
            DingTalkStreamClient,
            "open_connection",
            side_effect=RuntimeError("contains-sensitive-detail"),
        ):
            with self.assertRaisesRegex(RuntimeError, "contains-sensitive-detail"):
                client.open_connection()

        self.assertEqual(reporter.events, [("disconnected", "RuntimeError")])

    def test_start_records_connecting_and_disconnected_around_sdk_session(self) -> None:
        reporter = FakeReporter()
        client = MonitoredDingTalkStreamClient(Credential("id", "secret"), reporter)

        with patch.object(DingTalkStreamClient, "start", new=AsyncMock(return_value=None)):
            asyncio.run(client.start())

        self.assertEqual(
            reporter.events,
            [("connecting", ""), ("disconnected", "")],
        )

    def test_start_exception_keeps_the_exception_type_as_final_state(self) -> None:
        reporter = FakeReporter()
        client = MonitoredDingTalkStreamClient(Credential("id", "secret"), reporter)

        with patch.object(
            DingTalkStreamClient,
            "start",
            new=AsyncMock(side_effect=RuntimeError("sensitive-detail")),
        ):
            with self.assertRaisesRegex(RuntimeError, "sensitive-detail"):
                asyncio.run(client.start())

        self.assertEqual(
            reporter.events,
            [("connecting", ""), ("disconnected", "RuntimeError")],
        )


if __name__ == "__main__":
    unittest.main()
