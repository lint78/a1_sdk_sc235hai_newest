"""WebSocket 连接管理模块。"""
from __future__ import annotations

import logging

from fastapi import WebSocket

logger = logging.getLogger("ws")


class ConnectionManager:
    """维护所有活跃的 WebSocket 连接，并向它们广播 JSON 消息。"""

    def __init__(self) -> None:
        self._active: list[WebSocket] = []

    @property
    def active_count(self) -> int:
        return len(self._active)

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._active.append(websocket)
        logger.info("WebSocket 已连接，当前连接数=%d", len(self._active))

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self._active:
            self._active.remove(websocket)
            logger.info("WebSocket 已断开，当前连接数=%d", len(self._active))

    async def broadcast(self, message: dict) -> None:
        """向所有客户端广播，发送失败的连接会被清理。"""
        dead: list[WebSocket] = []
        for websocket in list(self._active):
            try:
                await websocket.send_json(message)
            except Exception:
                dead.append(websocket)
        for websocket in dead:
            self.disconnect(websocket)
