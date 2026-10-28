import json
import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from mumu_bot.runtime_state import StreamHealthReporter, read_json_object


class RuntimeStateTests(unittest.TestCase):
    def test_connected_state_is_atomic_and_contains_only_safe_fields(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "stream-health.json"
            reporter = StreamHealthReporter(
                path,
                "run-1",
                clock=lambda: datetime(2026, 8, 10, 9, 0, tzinfo=timezone.utc),
            )

            reporter.connected()

            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                {
                    "run_id": "run-1",
                    "state": "connected",
                    "updated_at": "2026-08-10T09:00:00+00:00",
                    "last_connected_at": "2026-08-10T09:00:00+00:00",
                    "last_heartbeat_at": "2026-08-10T09:00:00+00:00",
                    "error_type": "",
                },
            )
            self.assertFalse(path.with_suffix(".tmp").exists())

    def test_disconnected_preserves_last_successful_connection_time(self) -> None:
        times = iter(
            [
                datetime(2026, 8, 10, 9, 0, tzinfo=timezone.utc),
                datetime(2026, 8, 10, 9, 1, tzinfo=timezone.utc),
            ]
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "stream-health.json"
            reporter = StreamHealthReporter(path, "run-1", clock=lambda: next(times))

            reporter.connected()
            reporter.disconnected("ConnectionClosedError")

            self.assertEqual(
                read_json_object(path),
                {
                    "run_id": "run-1",
                    "state": "disconnected",
                    "updated_at": "2026-08-10T09:01:00+00:00",
                    "last_connected_at": "2026-08-10T09:00:00+00:00",
                    "last_heartbeat_at": "",
                    "error_type": "ConnectionClosedError",
                },
            )

    def test_invalid_or_non_object_json_is_treated_as_missing(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "stream-health.json"
            path.write_text("not-json", encoding="utf-8")
            self.assertIsNone(read_json_object(path))

            path.write_text("[]", encoding="utf-8")
            self.assertIsNone(read_json_object(path))


if __name__ == "__main__":
    unittest.main()
