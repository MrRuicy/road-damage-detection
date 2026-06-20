"""
实时检测 WebSocket 路由

前端通过 getUserMedia 采集摄像头画面，定时抓帧编码为 base64 经 WebSocket 发来，
后端推理后回传带框结果图与累计统计。
"""
import base64
import time

import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from services.detection_service import detection_service
from utils.file_utils import encode_image_to_base64
from core.config import settings

router = APIRouter()


def _decode_frame(data_url: str) -> np.ndarray | None:
    """将 base64 data URL 解码为 BGR 图像"""
    try:
        if "," in data_url:
            data_url = data_url.split(",", 1)[1]
        raw = base64.b64decode(data_url)
        arr = np.frombuffer(raw, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


@router.websocket("/ws/realtime")
async def realtime_detection(websocket: WebSocket):
    """
    实时检测 WebSocket。

    客户端发送 JSON：
      {"frame": "data:image/jpeg;base64,...", "conf": 0.5, "iou": 0.7, "model": "高精模型"}
    服务端回传 JSON：
      {"result_image": "...", "class_counts": {...}, "total_damages": N,
       "severity": "...", "severity_score": N, "fps": N}
    """
    await websocket.accept()

    # 会话累计统计
    cumulative_counts: dict = {}
    frame_count = 0
    session_start = time.time()
    last_ts = session_start

    try:
        while True:
            payload = await websocket.receive_json()

            frame_data = payload.get("frame")
            if not frame_data:
                continue

            conf = float(payload.get("conf", 0.5))
            iou = float(payload.get("iou", 0.7))
            model_name = payload.get("model", settings.DEFAULT_MODEL)

            image = _decode_frame(frame_data)
            if image is None:
                await websocket.send_json({"error": "无法解码图像帧"})
                continue

            image_area = float(image.shape[0] * image.shape[1])

            # 推理（复用已有同步检测逻辑）
            results, _ = await detection_service.detect_image(image, conf, iou, model_name)

            class_counts = detection_service.analyze_results(results)
            boxes = detection_service.extract_boxes(results)
            severity_score = detection_service.compute_severity_score(boxes, image_area)
            severity = detection_service.severity_from_score(severity_score)

            plotted = detection_service.plot_result(results)
            result_image = encode_image_to_base64(plotted)

            # 累计统计：每类病害取「单帧最大值」（峰值），避免同一病害逐帧累加刷高
            for k, v in class_counts.items():
                cumulative_counts[k] = max(cumulative_counts.get(k, 0), v)

            # 瞬时 FPS
            frame_count += 1
            now = time.time()
            inst_fps = 1.0 / (now - last_ts) if now > last_ts else 0.0
            last_ts = now

            await websocket.send_json({
                "result_image": result_image,
                "class_counts": class_counts,
                "cumulative_counts": cumulative_counts,
                "total_damages": sum(class_counts.values()),
                "severity": severity,
                "severity_score": severity_score,
                "fps": round(inst_fps, 1),
                "frame_count": frame_count,
            })

    except WebSocketDisconnect:
        # 客户端断开，正常结束
        pass
    except Exception as e:
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass
