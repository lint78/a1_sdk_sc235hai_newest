"""Publish alarm notifications to ntfy without blocking the MQTT callback."""
from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger("ntfy")


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class NtfyClient:
    """Send UTF-8 JSON notifications to ntfy in a background worker."""

    def __init__(self, server: str, topic: str, enabled: bool = True, click_url: str = "") -> None:
        self.server = server.rstrip("/")
        self.topic = topic.strip().strip("/")
        self.enabled = enabled and bool(self.topic)
        self.click_url = click_url
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ntfy")

    @classmethod
    def from_environment(cls) -> "NtfyClient":
        return cls(
            server=os.getenv("NTFY_SERVER", "https://ntfy.sh"),
            topic=os.getenv("NTFY_TOPIC", "home-alarm-100579-a1-7f3c9e2d"),
            enabled=_as_bool(os.getenv("NTFY_ENABLED"), default=True),
            click_url=os.getenv("NTFY_CLICK_URL", "https://alarm.100579.xyz"),
        )

    def publish_async(self, alarm: dict) -> None:
        if self.enabled:
            self._executor.submit(self._publish, dict(alarm))

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _publish(self, alarm: dict) -> None:
        alarm_type = str(alarm.get("alarm_type") or alarm.get("type") or "unknown")
        alarm_state = str(alarm.get("alarm_state") or alarm.get("state") or "unknown")
        device_id = str(alarm.get("device_id") or "unknown")
        frame = alarm.get("frame_index", alarm.get("frame"))
        type_label = {"fire": "火焰", "fall": "跌倒", "intrusion": "入侵"}.get(alarm_type, alarm_type)
        state_label = {"start": "开始", "end": "结束"}.get(alarm_state, alarm_state)

        payload = {
            "topic": self.topic,
            "title": f"家庭{type_label}报警",
            "message": (
                f"检测到{type_label}报警，状态：{state_label}。"
                f"设备：{device_id}，帧号：{frame if frame is not None else '-'}"
            ),
            # The ntfy JSON API expects a numeric priority from 1 to 5.
            "priority": 5 if alarm_state == "start" else 3,
            "tags": ["warning", alarm_type],
        }
        if self.click_url:
            payload["click"] = self.click_url

        request = Request(
            self.server,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            method="POST",
            headers={
                "Content-Type": "application/json; charset=utf-8",
                "Accept": "application/json",
                "User-Agent": "home-alarm-server/1.0",
            },
        )
        try:
            with urlopen(request, timeout=10) as response:
                response.read()
            logger.info("ntfy notification sent topic=%s type=%s state=%s device=%s", self.topic, alarm_type, alarm_state, device_id)
        except HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", errors="replace")
            except OSError:
                detail = ""
            logger.error(
                "ntfy request failed status=%s reason=%s detail=%s",
                exc.code,
                exc.reason,
                detail,
            )
        except (URLError, TimeoutError, OSError) as exc:
            logger.error("ntfy request failed: %s", exc)
