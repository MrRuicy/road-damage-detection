"""
视频检测 API 路由（异步任务模式）
"""
import asyncio
import threading

from fastapi import APIRouter, UploadFile, File, HTTPException, Form

from schemas.detection import TaskCreatedResponse, TaskStatusResponse
from services.task_manager import task_manager
from services.video_service import video_service
from services.record_service import record_service
from utils.file_utils import save_upload_file
from core.config import settings
from core.database import async_session_maker

router = APIRouter()


def _persist_video_result(result: dict, model_name: str, conf: float, iou: float):
    """视频处理完成回调：在后台线程里用独立事件循环写库"""
    async def _save():
        async with async_session_maker() as db:
            await record_service.create_record(
                db,
                detection_type="video",
                source_name=result["video_name"],
                model_name=model_name,
                conf_threshold=conf,
                iou_threshold=iou,
                total_damages=result["total_damages"],
                class_counts=result["class_counts"],
                severity=result["severity"],
                severity_score=result["severity_score"],
                result_image_path=result["result_video_url"],
                boxes=[],
                processing_time=result["processing_time"],
                extra={
                    "total_frames": result["total_frames"],
                    "processed_frames": result["processed_frames"],
                    "fps": result["fps"],
                },
            )

    asyncio.run(_save())


@router.post("/video", response_model=TaskCreatedResponse)
async def detect_video(
    file: UploadFile = File(...),
    conf_threshold: float = Form(0.5),
    iou_threshold: float = Form(0.7),
    model_name: str = Form(settings.DEFAULT_MODEL),
    frame_skip: int = Form(1),  # 追踪需较连续帧，默认仅跳 1 帧以保证 track_id 关联准确
    downscale: float = Form(1.0),
):
    """
    上传视频，启动后台检测任务，立即返回 task_id。
    前端通过 GET /video/task/{task_id} 轮询进度。
    """
    if not any(file.filename.lower().endswith(ext) for ext in settings.ALLOWED_VIDEO_EXTENSIONS):
        raise HTTPException(status_code=400, detail="不支持的视频格式")

    # 保存上传文件
    video_path = await save_upload_file(file, settings.UPLOAD_DIR)

    # 创建任务
    task = task_manager.create()

    # 后台线程处理（YOLO 推理为 CPU/GPU 密集型，放线程避免阻塞事件循环）
    def _run():
        video_service.process_video(
            task_id=task.id,
            video_path=video_path,
            source_name=file.filename,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            model_name=model_name,
            frame_skip=frame_skip,
            downscale=downscale,
            on_complete=lambda r: _persist_video_result(r, model_name, conf_threshold, iou_threshold),
        )

    threading.Thread(target=_run, daemon=True).start()

    return TaskCreatedResponse(task_id=task.id)


@router.get("/video/task/{task_id}", response_model=TaskStatusResponse)
async def get_video_task(task_id: str):
    """查询视频检测任务进度"""
    task = task_manager.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    return TaskStatusResponse(**task.to_dict())
