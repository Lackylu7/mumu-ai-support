from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path


def atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_json_object(path: Path) -> dict[str, object] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


class StreamHealthReporter:
    def __init__(
        self,
        path: Path,
        run_id: str,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.path = path
        self.run_id = run_id
        self.clock = clock or (lambda: datetime.now().astimezone())
        existing = read_json_object(path)
        self.last_connected_at = (
            str(existing.get("last_connected_at", ""))
            if existing is not None and existing.get("run_id") == run_id
            else ""
        )

    def starting(self) -> None:
        self._write("starting")

    def connecting(self) -> None:
        self._write("connecting")

    def connected(self) -> None:
        now = self.clock().isoformat()
        self.last_connected_at = now
        self._write("connected", now=now)

    def heartbeat(self) -> None:
        """Record a successful websocket keepalive without exposing payloads."""
        self._write("connected", now=self.clock().isoformat())

    def disconnected(self, error_type: str = "") -> None:
        self._write("disconnected", error_type=error_type)

    def _write(
        self,
        state: str,
        *,
        error_type: str = "",
        now: str | None = None,
    ) -> None:
        updated_at = now or self.clock().isoformat()
        atomic_write_json(
            self.path,
            {
                "run_id": self.run_id,
                "state": state,
                "updated_at": updated_at,
                "last_connected_at": self.last_connected_at,
                "last_heartbeat_at": updated_at if state == "connected" else "",
                "error_type": error_type,
            },
        )
