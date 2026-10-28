import json
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from mumu_bot.gaps import JsonGapRecorder


class JsonGapRecorderTests(unittest.TestCase):
    def test_repeated_normalized_question_increments_existing_gap(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "knowledge-gaps.json"
            recorder = JsonGapRecorder(path)
            first = datetime(2026, 8, 7, tzinfo=timezone.utc)

            recorder.record("怎么分红？", "知识库无匹配", first)
            recorder.record("怎么分红", "知识库无匹配", first + timedelta(hours=1))

            records = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["count"], 2)
            self.assertEqual(records[0]["first_seen"], "2026-08-07T00:00:00+00:00")
            self.assertEqual(records[0]["last_seen"], "2026-08-07T01:00:00+00:00")


if __name__ == "__main__":
    unittest.main()
