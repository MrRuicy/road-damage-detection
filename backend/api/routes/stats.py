"""
统计分析 API 路由
"""
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from services.stats_service import stats_service

router = APIRouter()


@router.get("/overview")
async def get_overview(db: AsyncSession = Depends(get_db)):
    """总览统计：累计/今日/本周/本月检测次数、病害总数、平均严重指数"""
    return await stats_service.overview(db)


@router.get("/class-distribution")
async def get_class_distribution(db: AsyncSession = Depends(get_db)):
    """病害类型分布"""
    return await stats_service.class_distribution(db)


@router.get("/severity-distribution")
async def get_severity_distribution(db: AsyncSession = Depends(get_db)):
    """严重度等级分布"""
    return await stats_service.severity_distribution(db)


@router.get("/trend")
async def get_trend(
    days: int = Query(7, ge=1, le=90),
    db: AsyncSession = Depends(get_db),
):
    """近 N 天检测趋势"""
    return await stats_service.trend(db, days=days)
