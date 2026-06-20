"""
视频检测服务

在后台线程中逐帧检测视频，实时更新任务进度，输出带框结果视频。
"""
import os
import time
import uuid
import shutil
import subprocess
from datetime import datetime
from typing import Optional

import cv2

from core.config import settings
from models.yolo_model import model_manager
from services.detection_service import detection_service
from services.task_manager import task_manager, TaskStatus


def _transcode_to_h264(src_path: str) -> bool:
    """
    用 ffmpeg 将视频转码为浏览器可直接播放的 H.264 (yuv420p)。
    成功则原地替换 src_path，返回 True；ffmpeg 不存在或失败返回 False（保留原文件）。
    """
    if shutil.which("ffmpeg") is None:
        return False
    tmp_out = src_path + ".h264.mp4"
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", src_path,
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",  # web 播放优化：moov 前置
                "-preset", "veryfast",
                tmp_out,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=300,
        )
        os.replace(tmp_out, src_path)
        return True
    except Exception:
        if os.path.exists(tmp_out):
            try:
                os.remove(tmp_out)
            except OSError:
                pass
        return False


class VideoService:
    """视频逐帧检测"""

    @staticmethod
    def process_video(
        task_id: str,
        video_path: str,
        source_name: str,
        conf_threshold: float,
        iou_threshold: float,
        model_name: str,
        frame_skip: int = 2,
        downscale: float = 1.0,
        on_complete=None,
    ):
        """
        后台线程入口：逐帧检测并写出结果视频。

        frame_skip: 每隔多少帧检测一次（降低开销）
        downscale: 推理前缩放比例（加速）
        on_complete: 完成回调 (result_dict) -> None，用于写库
        """
        try:
            task_manager.update(task_id, status=TaskStatus.RUNNING, message="正在读取视频...")

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError("无法打开视频文件")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            task_manager.update(task_id, total_frames=total_frames)

            # 输出视频
            os.makedirs(settings.RESULT_DIR, exist_ok=True)
            out_name = f"video_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}.mp4"
            out_path = os.path.join(settings.RESULT_DIR, out_name)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

            model = model_manager.get_model(model_name)

            total_class_counts: dict = {}
            peak_boxes: list = []  # 病害数最多那一帧的检测框（作为严重度代表帧）
            peak_frame_damages = -1
            frame_idx = 0
            processed = 0
            start = time.time()
            last_plotted = None

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1

                # 跳帧：非检测帧复用上一帧的检测画面
                if frame_skip > 0 and (frame_idx % (frame_skip + 1) != 0):
                    writer.write(last_plotted if last_plotted is not None else frame)
                    continue

                infer_frame = frame
                if downscale and downscale < 1.0:
                    infer_frame = cv2.resize(frame, None, fx=downscale, fy=downscale)

                results = model.predict(infer_frame, conf=conf_threshold, iou=iou_threshold, verbose=False)
                plotted = results[0].plot()
                if plotted.shape[:2] != (height, width):
                    plotted = cv2.resize(plotted, (width, height))
                last_plotted = plotted
                writer.write(plotted)

                # 统计：每类病害取「单帧最大值」（峰值），避免同一病害逐帧累加
                counts = detection_service.analyze_results(results)
                for k, v in counts.items():
                    total_class_counts[k] = max(total_class_counts.get(k, 0), v)

                # 记录病害最多的那一帧的检测框，作为严重度评估的代表帧
                frame_boxes = detection_service.extract_boxes(results)
                frame_damages = len(frame_boxes)
                if frame_damages > peak_frame_damages:
                    peak_frame_damages = frame_damages
                    peak_boxes = frame_boxes

                processed += 1
                if total_frames > 0:
                    progress = min(frame_idx / total_frames * 100, 99.0)
                    task_manager.update(
                        task_id,
                        progress=round(progress, 1),
                        processed_frames=processed,
                        message=f"检测中... {frame_idx}/{total_frames} 帧",
                    )

            cap.release()
            writer.release()

            # 转码为 H.264，让浏览器可直接在线播放（ffmpeg 不可用时降级保留 mp4v）
            task_manager.update(task_id, message="正在转码输出视频...")
            _transcode_to_h264(out_path)

            processing_time = time.time() - start
            severity_score = detection_service.compute_severity_score(peak_boxes)
            severity = detection_service.severity_from_score(severity_score)
            total_damages = sum(total_class_counts.values())

            result = {
                "video_name": source_name,
                "result_video_url": f"/static/results/{out_name}",
                "total_frames": total_frames,
                "processed_frames": processed,
                "fps": round(fps, 1),
                "total_damages": total_damages,
                "class_counts": total_class_counts,
                "severity": severity,
                "severity_score": severity_score,
                "processing_time": round(processing_time, 2),
            }

            # 清理上传的原始视频
            if os.path.exists(video_path):
                try:
                    os.remove(video_path)
                except OSError:
                    pass

            if on_complete:
                on_complete(result)

            task_manager.update(
                task_id,
                status=TaskStatus.COMPLETED,
                progress=100.0,
                message="检测完成",
                result=result,
            )

        except Exception as e:
            task_manager.update(
                task_id,
                status=TaskStatus.FAILED,
                error=str(e),
                message=f"检测失败: {e}",
            )
            if os.path.exists(video_path):
                try:
                    os.remove(video_path)
                except OSError:
                    pass


video_service = VideoService()
