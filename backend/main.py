"""
FastAPI 主应用入口
"""
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
import uvicorn

from api.routes import detection, system, records, stats, video, realtime, export
from core.config import settings
from core.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动：初始化数据库、确保目录存在
    await init_db()
    os.makedirs(settings.DATA_DIR, exist_ok=True)  # 数据库等持久数据
    os.makedirs(settings.RESULT_DIR, exist_ok=True)
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    print("✓ 数据库与目录已就绪")
    yield
    # 关闭：暂无清理工作


app = FastAPI(
    title=settings.PROJECT_NAME,
    description="基于 YOLO 的道路病害检测系统 API",
    version=settings.VERSION,
    lifespan=lifespan,
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件服务（带框结果图/结果视频）
from core.config import BACKEND_DIR
STATIC_DIR = os.path.join(BACKEND_DIR, "static")
os.makedirs(settings.RESULT_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# 注册路由
app.include_router(detection.router, prefix="/api/detection", tags=["检测"])
app.include_router(video.router, prefix="/api/detection", tags=["视频检测"])
app.include_router(system.router, prefix="/api/system", tags=["系统"])
app.include_router(records.router, prefix="/api/records", tags=["历史记录"])
app.include_router(stats.router, prefix="/api/stats", tags=["统计分析"])
app.include_router(realtime.router, prefix="/api/detection", tags=["实时检测"])
app.include_router(export.router, prefix="/api/export", tags=["导出"])


# ===== 条件性前端托管（容器化/生产模式） =====
# 当 frontend/dist 存在时（npm run build 已执行），FastAPI 同时托管前端静态文件
# 开发模式（frontend/dist 不存在）：后端仅作 API，前端独立运行在 5173 端口
# 生产/容器模式（frontend/dist 存在）：后端 8000 端口同时提供前端和 API
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)
FRONTEND_DIST = os.path.join(PROJECT_ROOT, "frontend", "dist")


class SPAStaticFiles(StaticFiles):
    """支持 Vue Router history 模式：未匹配到的路径回退到 index.html"""
    async def get_response(self, path: str, scope):
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as ex:
            # 前端路由（如 /dashboard）不是真实文件，StaticFiles 抛出 404，
            # 回退给 index.html 由 Vue Router 接管
            if ex.status_code == 404:
                return await super().get_response("index.html", scope)
            raise


# API 健康检查端点（两种模式均可用，注册在前端挂载之前）
@app.get("/api/health")
async def health():
    return {"status": "ok", "version": settings.VERSION}


if os.path.exists(FRONTEND_DIST):
    print(f"✓ 检测到前端构建产物，启用单端口模式（前端+API）")
    # SPA 挂载到根路径，必须在所有 API 路由注册之后（API 优先匹配）
    app.mount("/", SPAStaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
else:
    print("ℹ 前端构建产物不存在，仅提供 API 服务（开发模式）")

    @app.get("/")
    async def root():
        return {
            "message": "道路病害检测系统 API",
            "version": settings.VERSION,
            "docs": "/docs",
            "mode": "development"
        }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
