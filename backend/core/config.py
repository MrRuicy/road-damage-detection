"""
应用配置
"""
import os
from typing import List
from pydantic_settings import BaseSettings

# 锚定 backend 目录，所有相对路径基于此，避免依赖运行时工作目录
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)


class Settings(BaseSettings):
    # 基础配置
    PROJECT_NAME: str = "道路病害检测系统"
    VERSION: str = "2.0.0"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True

    # CORS 配置
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
    ]

    # 模型配置（绝对路径，权重位于 backend/weights/）
    # 高召回模型：漏检少，适合普查初筛（实测 mAP50=0.861, 召回率=0.840）
    # 高精确模型：误检少，适合复核确认（实测 mAP50=0.858, 精确率=0.865）
    MODEL_PATHS: dict = {
        "高召回模型": os.path.join(BACKEND_DIR, "weights", "road_damage_high_recall.pt"),
        "高精确模型": os.path.join(BACKEND_DIR, "weights", "road_damage_high_precision.onnx"),
    }
    DEFAULT_MODEL: str = "高召回模型"

    # 检测参数
    DEFAULT_CONF_THRESHOLD: float = 0.5
    DEFAULT_IOU_THRESHOLD: float = 0.7

    # 文件上传配置（绝对路径，基于 backend 目录）
    UPLOAD_DIR: str = os.path.join(BACKEND_DIR, "uploads")
    RESULT_DIR: str = os.path.join(BACKEND_DIR, "static", "results")  # 带框结果图/视频持久化目录
    DATA_DIR: str = os.path.join(BACKEND_DIR, "data")  # 数据库等持久数据，方便容器挂载
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB
    ALLOWED_IMAGE_EXTENSIONS: List[str] = [".jpg", ".jpeg", ".png"]
    ALLOWED_VIDEO_EXTENSIONS: List[str] = [".mp4", ".avi", ".mov"]

    # 数据库配置（SQLite 单文件，统一放 data/ 子目录方便容器挂载）
    DATABASE_URL: str = f"sqlite+aiosqlite:///{os.path.join(BACKEND_DIR, 'data', 'road_damage.db')}"

    # 类别配置
    CLASS_NAMES: dict = {
        0: "纵向裂缝",
        1: "横向裂缝",
        2: "块状裂缝",
        3: "坑洼",
        4: "修补"
    }

    # 病害严重度权重（用于加权病害指数，坑洼最严重，修补几乎无害）
    CLASS_WEIGHTS: dict = {
        "纵向裂缝": 1.0,
        "横向裂缝": 1.2,
        "块状裂缝": 1.5,
        "坑洼": 2.0,
        "修补": 0.3,
    }

    class Config:
        case_sensitive = True


settings = Settings()
