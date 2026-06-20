"""
后端启动脚本
"""
import uvicorn
from core.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        # 排除上传/临时目录，避免保存文件时触发热重载中断请求
        reload_excludes=["uploads/*", "temp/*", "*.jpg", "*.png", "*.mp4"],
    )
