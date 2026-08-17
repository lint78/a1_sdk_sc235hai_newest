"""家庭报警服务：FastAPI + MQTT 订阅 + WebSocket 推送。"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from starlette.websockets import WebSocketDisconnect

import db
from models import AlarmMessage
from mqtt_client import MqttAlarmClient
from ntfy_client import NtfyClient
from ws_manager import ConnectionManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("app")

MQTT_HOST = "broker.emqx.io"
MQTT_PORT = 1883
MQTT_USERNAME = ""
MQTT_PASSWORD = ""
MQTT_TOPIC = "home/a1-001/alarm"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """服务启动时初始化数据库并启动 MQTT 客户端，关闭时停止。"""
    db.init_db()
    manager = ConnectionManager()
    ntfy_client = NtfyClient.from_environment()
    loop = asyncio.get_running_loop()

    def on_alarm(data: dict) -> None:
        """MQTT 线程回调：校验 -> 入库 -> 通过主事件循环广播。"""
        try:
            message = AlarmMessage.model_validate(data)
        except Exception as exc:
            logger.error("告警数据校验失败，已跳过: %s data=%s", exc, data)
            return

        try:
            record = db.insert_event(message.model_dump())
        except Exception:
            logger.exception("写入 SQLite 失败")
            return

        if record:
            logger.info(
                "已入库并准备推送: id=%s device=%s type=%s state=%s",
                record["id"],
                record["device_id"],
                record["alarm_type"],
                record["alarm_state"],
            )
            asyncio.run_coroutine_threadsafe(manager.broadcast(record), loop)
            ntfy_client.publish_async(record)

    mqtt_client = MqttAlarmClient(
        host=MQTT_HOST,
        port=MQTT_PORT,
        username=MQTT_USERNAME,
        password=MQTT_PASSWORD,
        topic=MQTT_TOPIC,
        on_alarm=on_alarm,
    )
    mqtt_client.start()

    app.state.ws_manager = manager
    app.state.mqtt_client = mqtt_client
    app.state.ntfy_client = ntfy_client

    logger.info("服务启动完成，监听 topic=%s", MQTT_TOPIC)
    yield

    mqtt_client.stop()
    ntfy_client.close()
    logger.info("服务已关闭")


app = FastAPI(title="Home Alarm Server", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
def index():
    html_path = Path(__file__).resolve().parent / "templates" / "index.html"
    # 用 utf-8-sig 读取：兼容带 BOM 的文件，并自动剥掉 BOM，避免返回给浏览器时出现多余字符。
    return HTMLResponse(html_path.read_text(encoding="utf-8-sig"))


@app.get("/api/health")
def health():
    mqtt_client = getattr(app.state, "mqtt_client", None)
    return {
        "status": "ok",
        "mqtt_connected": mqtt_client.is_connected() if mqtt_client else False,
    }


@app.get("/api/alarms")
def list_alarms(limit: int = 100):
    limit = max(1, min(limit, 1000))
    return db.get_recent_alarms(limit)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    manager = app.state.ws_manager
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)
