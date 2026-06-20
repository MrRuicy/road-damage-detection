"""
数据库连接与会话管理（SQLAlchemy 2.0 异步）
"""
import os

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase

from core.config import settings


class Base(DeclarativeBase):
    """ORM 模型基类"""
    pass


# 确保数据库目录存在（SQLite 不会自动创建父目录），避免首次连接报错
os.makedirs(settings.DATA_DIR, exist_ok=True)

# 异步引擎
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    future=True,
)

# 异步会话工厂
async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def init_db():
    """初始化数据库，创建所有表"""
    # 导入模型以注册到 Base.metadata
    from models import db_models  # noqa: F401

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db() -> AsyncSession:
    """FastAPI 依赖：提供数据库会话"""
    async with async_session_maker() as session:
        yield session
