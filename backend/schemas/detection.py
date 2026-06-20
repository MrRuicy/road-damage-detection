"""
数据模型定义
"""
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class DetectionBox(BaseModel):
    """检测框"""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str


class DetectionResult(BaseModel):
    """单张图片检测结果"""
    image_name: str
    boxes: List[DetectionBox]
    class_counts: Dict[str, int]
    total_damages: int
    severity: str
    severity_score: float = 0.0  # 0~100 加权病害指数
    processing_time: float
    result_image: Optional[str] = None  # base64 编码的带框结果图
    record_id: Optional[int] = None  # 入库后的记录 ID
    is_duplicate: bool = False  # 是否命中去重（复用了已有检测结果）


class BatchDetectionResult(BaseModel):
    """批量检测结果"""
    total_images: int
    total_damages: int
    overall_severity: str
    overall_severity_score: float = 0.0
    class_counts: Dict[str, int]
    results: List[DetectionResult]


class VideoDetectionResult(BaseModel):
    """视频检测结果"""
    video_name: str
    total_frames: int
    processed_frames: int
    total_damages: int
    severity: str
    max_severity: str
    class_counts: Dict[str, int]
    fps: float
    duration: float
    result_video_url: Optional[str] = None


class TaskCreatedResponse(BaseModel):
    """异步任务创建响应"""
    task_id: str
    message: str = "任务已创建"


class TaskStatusResponse(BaseModel):
    """异步任务状态"""
    id: str
    status: str
    progress: float
    message: str
    total_frames: int
    processed_frames: int
    result: Optional[dict] = None
    error: Optional[str] = None


class DetectionConfig(BaseModel):
    """检测配置"""
    model_name: str = "高召回模型"
    conf_threshold: float = Field(default=0.5, ge=0.1, le=1.0)
    iou_threshold: float = Field(default=0.7, ge=0.1, le=1.0)
    selected_classes: Optional[List[str]] = None


class SystemStatus(BaseModel):
    """系统状态"""
    cpu_usage: float
    memory_usage: float
    gpu_usage: float
    gpu_memory: Optional[float] = None
    current_model: str
    available_models: List[str]


class RecordItem(BaseModel):
    """历史记录条目"""
    id: int
    detection_type: str
    source_name: str
    model_name: str
    conf_threshold: float
    iou_threshold: float
    total_damages: int
    class_counts: Dict[str, int]
    severity: str
    severity_score: float
    result_image_path: Optional[str] = None
    boxes: List[dict] = []
    processing_time: float
    created_at: Optional[str] = None
    extra: Optional[dict] = None


class RecordListResponse(BaseModel):
    """分页记录列表"""
    total: int
    page: int
    page_size: int
    items: List[RecordItem]
