"""数据模型定义。"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class AlarmMessage(BaseModel):
    """ESP8266 上报的告警消息结构，字段名与 MQTT 消息 JSON 保持一致。"""

    device_id: str = Field(..., description="设备编号，如 a1-001")
    type: str = Field(..., description="告警类型：fire / intrusion / fall 等")
    state: str = Field(..., description="告警状态：start / end 等")
    frame: Optional[int] = Field(default=None, description="帧序号")
    source: Optional[str] = Field(default=None, description="来源，如 a1")
    ts_millis: Optional[int] = Field(default=None, description="设备侧时间戳（毫秒）")
