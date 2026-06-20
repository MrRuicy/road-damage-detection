"""
检测记录持久化服务
"""
import os
import uuid
from datetime import datetime
from typing import Optional, List

import cv2
import numpy as np
from sqlalchemy import select, func, delete
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import settings
from models.db_models import DetectionRecord


class RecordService:
    """检测记录的存储与查询"""

    @staticmethod
    def save_result_image(image: np.ndarray, prefix: str = "result") -> str:
        """
        将带框结果图保存到静态目录，返回相对 URL 路径。
        """
        os.makedirs(settings.RESULT_DIR, exist_ok=True)
        filename = f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}.jpg"
        filepath = os.path.join(settings.RESULT_DIR, filename)
        cv2.imwrite(filepath, image)
        # 返回可被前端访问的静态 URL
        return f"/static/results/{filename}"

    @staticmethod
    async def create_record(
        db: AsyncSession,
        *,
        detection_type: str,
        source_name: str,
        model_name: str,
        conf_threshold: float,
        iou_threshold: float,
        total_damages: int,
        class_counts: dict,
        severity: str,
        severity_score: float,
        result_image_path: Optional[str] = None,
        boxes: Optional[list] = None,
        processing_time: float = 0.0,
        extra: Optional[dict] = None,
        content_hash: Optional[str] = None,
    ) -> DetectionRecord:
        """创建并持久化一条检测记录"""
        record = DetectionRecord(
            detection_type=detection_type,
            source_name=source_name,
            model_name=model_name,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            total_damages=total_damages,
            class_counts=class_counts,
            severity=severity,
            severity_score=severity_score,
            result_image_path=result_image_path,
            boxes=boxes or [],
            processing_time=processing_time,
            extra=extra,
            content_hash=content_hash,
        )
        db.add(record)
        await db.commit()
        await db.refresh(record)
        return record

    @staticmethod
    async def find_duplicate(
        db: AsyncSession,
        *,
        content_hash: str,
        model_name: str,
        conf_threshold: float,
        iou_threshold: float,
    ) -> Optional[DetectionRecord]:
        """
        查找内容、模型、参数完全一致的已有记录（用于智能去重）。
        命中则可直接复用旧结果，避免重复检测与入库。
        """
        stmt = (
            select(DetectionRecord)
            .where(
                DetectionRecord.content_hash == content_hash,
                DetectionRecord.model_name == model_name,
                DetectionRecord.conf_threshold == conf_threshold,
                DetectionRecord.iou_threshold == iou_threshold,
            )
            .order_by(DetectionRecord.created_at.desc())
            .limit(1)
        )
        return (await db.execute(stmt)).scalars().first()

    @staticmethod
    async def list_records(
        db: AsyncSession,
        *,
        page: int = 1,
        page_size: int = 10,
        detection_type: Optional[str] = None,
        severity: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> tuple[List[DetectionRecord], int]:
        """分页查询检测记录，支持类型/严重度/日期筛选"""
        conditions = []
        if detection_type:
            conditions.append(DetectionRecord.detection_type == detection_type)
        if severity:
            conditions.append(DetectionRecord.severity == severity)
        if start_date:
            conditions.append(DetectionRecord.created_at >= start_date)
        if end_date:
            conditions.append(DetectionRecord.created_at <= end_date)

        # 总数
        count_stmt = select(func.count(DetectionRecord.id))
        if conditions:
            count_stmt = count_stmt.where(*conditions)
        total = (await db.execute(count_stmt)).scalar() or 0

        # 分页数据
        stmt = select(DetectionRecord)
        if conditions:
            stmt = stmt.where(*conditions)
        stmt = (
            stmt.order_by(DetectionRecord.created_at.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        records = (await db.execute(stmt)).scalars().all()
        return list(records), total

    @staticmethod
    async def get_record(db: AsyncSession, record_id: int) -> Optional[DetectionRecord]:
        """按 ID 获取单条记录"""
        return await db.get(DetectionRecord, record_id)

    @staticmethod
    async def delete_record(db: AsyncSession, record_id: int) -> bool:
        """删除记录，同时清理结果图文件"""
        record = await db.get(DetectionRecord, record_id)
        if not record:
            return False

        # 清理磁盘上的结果图
        if record.result_image_path:
            fname = os.path.basename(record.result_image_path)
            fpath = os.path.join(settings.RESULT_DIR, fname)
            if os.path.exists(fpath):
                try:
                    os.remove(fpath)
                except OSError:
                    pass

        await db.delete(record)
        await db.commit()
        return True


record_service = RecordService()
