"""
导出 API 路由：PDF 报告 + Excel 记录
"""
from datetime import datetime
from typing import Optional
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from services.record_service import record_service
from services.export_service import export_service

router = APIRouter()


@router.get("/pdf/{record_id}")
async def export_pdf(record_id: int, db: AsyncSession = Depends(get_db)):
    """导出单条记录的 PDF 报告"""
    record = await record_service.get_record(db, record_id)
    if not record:
        raise HTTPException(status_code=404, detail="记录不存在")

    pdf_bytes = export_service.generate_pdf(record)
    filename = quote(f"检测报告_{record.id}.pdf")
    return StreamingResponse(
        iter([pdf_bytes]),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{filename}"},
    )


@router.get("/records.xlsx")
async def export_records_excel(
    detection_type: Optional[str] = None,
    severity: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    db: AsyncSession = Depends(get_db),
):
    """导出检测记录为 Excel（支持与历史列表相同的筛选条件）"""
    # 取全部匹配记录（上限 10000 条）
    records, _ = await record_service.list_records(
        db,
        page=1,
        page_size=10000,
        detection_type=detection_type,
        severity=severity,
        start_date=start_date,
        end_date=end_date,
    )

    xlsx_bytes = export_service.generate_excel(records)
    filename = quote(f"检测记录_{datetime.now():%Y%m%d_%H%M%S}.xlsx")
    return StreamingResponse(
        iter([xlsx_bytes]),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{filename}"},
    )
