from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol


class GapRecorder(Protocol):
    def record(self, question: str, reason: str, now: datetime) -> None: ...


def _normalize(question: str) -> str:
    return re.sub(r"[\s，。！？、,.!?]+", "", question).lower()


@dataclass
class JsonGapRecorder:
    path: Path

    def record(self, question: str, reason: str, now: datetime) -> None:
        records = self._load()
        normalized = _normalize(question)
        for record in records:
            if record.get("normalized_question") == normalized:
                record["count"] = int(record.get("count", 0)) + 1
                record["last_seen"] = now.isoformat()
                record["reason"] = reason
                self._write(records)
                return
        records.append(
            {
                "question": question,
                "normalized_question": normalized,
                "reason": reason,
                "count": 1,
                "first_seen": now.isoformat(),
                "last_seen": now.isoformat(),
                "status": "待处理",
            }
        )
        self._write(records)

    def _load(self) -> list[dict[str, object]]:
        if not self.path.exists():
            return []
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("知识缺口文件无法读取") from error
        if not isinstance(payload, list) or not all(
            isinstance(record, dict) for record in payload
        ):
            raise ValueError("知识缺口文件格式无效")
        return payload

    def _write(self, records: list[dict[str, object]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(f"{self.path.suffix}.tmp")
        temporary.write_text(
            json.dumps(records, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)
