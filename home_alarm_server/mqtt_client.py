"""MQTT 客户端模块：订阅 home/a1-001/alarm 并回调处理。"""
from __future__ import annotations

import json
import logging
from typing import Callable

import paho.mqtt.client as mqtt

logger = logging.getLogger("mqtt")

# 收到一条解析成功的告警 dict 后回调。
AlarmCallback = Callable[[dict], None]


class MqttAlarmClient:
    """封装 paho-mqtt，负责连接、订阅、解析，并调用 on_alarm。"""

    def __init__(
        self,
        host: str,
        port: int,
        username: str,
        password: str,
        topic: str,
        on_alarm: AlarmCallback,
    ) -> None:
        self.host = host
        self.port = port
        self.topic = topic
        self.on_alarm = on_alarm

        self._client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id="home-alarm-server",
        )
        self._client.username_pw_set(username, password)
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message

    def start(self) -> None:
        logger.info("正在连接 MQTT broker %s:%d ...", self.host, self.port)
        self._client.connect(self.host, self.port, keepalive=60)
        self._client.loop_start()

    def stop(self) -> None:
        self._client.loop_stop()
        self._client.disconnect()
        logger.info("MQTT 客户端已停止")

    def is_connected(self) -> bool:
        return self._client.is_connected()

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        # paho-mqtt 2.x 的 reason_code 是 ReasonCode 对象，兼容判断 is_failure。
        is_failure = getattr(reason_code, "is_failure", reason_code != 0)
        if not is_failure:
            logger.info("MQTT 已连接，订阅 topic: %s", self.topic)
            client.subscribe(self.topic, qos=1)
        else:
            logger.error("MQTT 连接失败，reason_code=%s", reason_code)

    def _on_disconnect(self, client, userdata, disconnect_flags, reason_code, properties):
        logger.warning("MQTT 连接断开，reason_code=%s", reason_code)

    def _on_message(self, client, userdata, msg):
        try:
            text = msg.payload.decode("utf-8")
        except UnicodeDecodeError:
            logger.error("消息不是合法 UTF-8，已跳过: topic=%s", msg.topic)
            return

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            logger.error(
                "JSON 解析失败，已跳过: topic=%s error=%s payload=%s",
                msg.topic,
                exc,
                text,
            )
            return

        if not isinstance(data, dict):
            logger.error("消息不是 JSON 对象，已跳过: topic=%s payload=%s", msg.topic, text)
            return

        logger.info("收到告警: topic=%s data=%s", msg.topic, json.dumps(data, ensure_ascii=False))
        try:
            self.on_alarm(data)
        except Exception:
            logger.exception("处理告警消息时发生异常")
