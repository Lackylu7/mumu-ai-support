from __future__ import annotations

import asyncio
import logging
from typing import Protocol

from dingtalk_stream import Credential, DingTalkStreamClient


logger = logging.getLogger(__name__)


class StreamHealthPort(Protocol):
    def connecting(self) -> None: ...

    def connected(self) -> None: ...

    def heartbeat(self) -> None: ...

    def disconnected(self, error_type: str = "") -> None: ...


class MonitoredDingTalkStreamClient(DingTalkStreamClient):
    def __init__(self, credential: Credential, reporter: StreamHealthPort) -> None:
        super().__init__(credential)
        self.reporter = reporter

    def open_connection(self):
        logger.warning(
            "stream_callback_topics_registered topics=%s",
            sorted(self.callback_handler_map),
        )
        try:
            result = super().open_connection()
        except Exception as error:
            self.reporter.disconnected(type(error).__name__)
            raise
        if result is None:
            self.reporter.disconnected("OpenConnectionFailed")
        else:
            self.reporter.connected()
        return result

    async def keepalive(self, ws, ping_interval=60):
        """Keep the supervisor informed while the SDK socket is still alive."""
        while True:
            await asyncio.sleep(ping_interval)
            try:
                await ws.ping()
                self.reporter.heartbeat()
            except Exception as error:
                self.reporter.disconnected(type(error).__name__)
                try:
                    await ws.close()
                except Exception:
                    pass
                return

    async def start(self):
        self.reporter.connecting()
        try:
            result = await super().start()
        except BaseException as error:
            self.reporter.disconnected(type(error).__name__)
            raise
        self.reporter.disconnected()
        return result
