"""
历史记录 API 路由
"""
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from schemas.detection import RecordListResponse, RecordItem
from services.record_service import record_service

router = APIRouter()


@router.get("", response_model=RecordListResponse)
async def list_records(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    detection_type: Optional[str] = None,
    severity: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    db: AsyncSession = Depends(get_db),
):
    """分页查询检测历史，支持类型/严重度/日期筛选"""
    records, total = await record_service.list_records(
        db,
        page=page,
        page_size=page_size,
        detection_type=detection_type,
        severity=severity,
        start_date=start_date,
        end_date=end_date,
    )
    return RecordListResponse(
        total=total,
        page=page,
        page_size=page_size,
        items=[RecordItem(**r.to_dict()) for r in records],
    )


@router.get("/{record_id}", response_model=RecordItem)
async def get_record(record_id: int, db: AsyncSession = Depends(get_db)):
    """获取单条记录详情"""
    record = await record_service.get_record(db, record_id)
    if not record:
        raise HTTPException(status_code=404, detail="记录不存在")
    return RecordItem(**record.to_dict())


@router.delete("/{record_id}")
async def delete_record(record_id: int, db: AsyncSession = Depends(get_db)):
    """删除记录及其结果图"""
    ok = await record_service.delete_record(db, record_id)
    if not ok:
        raise HTTPException(status_code=404, detail="记录不存在")
    return {"message": "已删除", "id": record_id}
