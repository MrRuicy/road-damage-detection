"""
ORM 数据模型
"""
from datetime import datetime
from sqlalchemy import String, Integer, Float, DateTime, JSON, Text
from sqlalchemy.orm import Mapped, mapped_column

from core.database import Base


class DetectionRecord(Base):
    """检测记录表"""
    __tablename__ = "detection_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # 检测元信息
    detection_type: Mapped[str] = mapped_column(String(20), index=True)  # image / batch / video / realtime
    source_name: Mapped[str] = mapped_column(String(255))  # 原始文件名
    model_name: Mapped[str] = mapped_column(String(50))
    content_hash: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)  # 图片内容MD5，用于去重

    # 检测参数
    conf_threshold: Mapped[float] = mapped_column(Float, default=0.5)
    iou_threshold: Mapped[float] = mapped_column(Float, default=0.7)

    # 检测结果
    total_damages: Mapped[int] = mapped_column(Integer, default=0)
    class_counts: Mapped[dict] = mapped_column(JSON, default=dict)  # {"坑洼": 3, ...}
    severity: Mapped[str] = mapped_column(String(20), default="无病害", index=True)
    severity_score: Mapped[float] = mapped_column(Float, default=0.0)  # 0~100 加权指数

    # 结果产物
    result_image_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    boxes: Mapped[list] = mapped_column(JSON, default=list)  # 检测框明细

    # 性能与时间
    processing_time: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now, index=True
    )

    # 备注（视频帧数等附加信息）
    extra: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "detection_type": self.detection_type,
            "source_name": self.source_name,
            "model_name": self.model_name,
            "conf_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold,
            "total_damages": self.total_damages,
            "class_counts": self.class_counts,
            "severity": self.severity,
            "severity_score": self.severity_score,
            "result_image_path": self.result_image_path,
            "boxes": self.boxes,
            "processing_time": self.processing_time,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "extra": self.extra,
        }
