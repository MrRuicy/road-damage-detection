"""
统计分析服务
"""
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from models.db_models import DetectionRecord


class StatsService:
    """检测数据的聚合统计"""

    @staticmethod
    async def overview(db: AsyncSession) -> dict:
        """总览：累计检测次数、病害总数、今日/本周/本月检测次数、平均严重指数"""
        now = datetime.now()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        week_start = today_start - timedelta(days=now.weekday())
        month_start = today_start.replace(day=1)

        async def count_since(dt: Optional[datetime]) -> int:
            stmt = select(func.count(DetectionRecord.id))
            if dt is not None:
                stmt = stmt.where(DetectionRecord.created_at >= dt)
            return (await db.execute(stmt)).scalar() or 0

        total = await count_since(None)
        today = await count_since(today_start)
        week = await count_since(week_start)
        month = await count_since(month_start)

        total_damages = (
            await db.execute(select(func.coalesce(func.sum(DetectionRecord.total_damages), 0)))
        ).scalar() or 0

        avg_score = (
            await db.execute(select(func.coalesce(func.avg(DetectionRecord.severity_score), 0.0)))
        ).scalar() or 0.0

        return {
            "total_detections": total,
            "today_detections": today,
            "week_detections": week,
            "month_detections": month,
            "total_damages": int(total_damages),
            "avg_severity_score": round(float(avg_score), 1),
        }

    @staticmethod
    async def class_distribution(db: AsyncSession) -> dict:
        """病害类型分布：累加所有记录的 class_counts"""
        rows = (await db.execute(select(DetectionRecord.class_counts))).scalars().all()
        dist: dict[str, int] = {}
        for cc in rows:
            if not cc:
                continue
            for name, cnt in cc.items():
                dist[name] = dist.get(name, 0) + int(cnt)
        return dist

    @staticmethod
    async def severity_distribution(db: AsyncSession) -> dict:
        """严重度等级分布"""
        stmt = select(
            DetectionRecord.severity, func.count(DetectionRecord.id)
        ).group_by(DetectionRecord.severity)
        rows = (await db.execute(stmt)).all()
        return {severity: count for severity, count in rows}

    @staticmethod
    async def trend(db: AsyncSession, days: int = 7) -> list[dict]:
        """近 N 天检测次数与病害数趋势（按天聚合）"""
        now = datetime.now()
        start = (now - timedelta(days=days - 1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )

        stmt = (
            select(
                func.date(DetectionRecord.created_at).label("day"),
                func.count(DetectionRecord.id).label("detections"),
                func.coalesce(func.sum(DetectionRecord.total_damages), 0).label("damages"),
            )
            .where(DetectionRecord.created_at >= start)
            .group_by(func.date(DetectionRecord.created_at))
        )
        rows = (await db.execute(stmt)).all()
        by_day = {str(r.day): {"detections": r.detections, "damages": int(r.damages)} for r in rows}

        # 补齐缺失的日期，保证前端折线连续
        result = []
        for i in range(days):
            day = (start + timedelta(days=i)).strftime("%Y-%m-%d")
            entry = by_day.get(day, {"detections": 0, "damages": 0})
            result.append({"date": day, **entry})
        return result


stats_service = StatsService()
