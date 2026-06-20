"""
系统相关 API 路由
"""
from fastapi import APIRouter, HTTPException
import psutil
import GPUtil

from schemas.detection import SystemStatus
from models.yolo_model import model_manager

router = APIRouter()


@router.get("/status", response_model=SystemStatus)
async def get_system_status():
    """
    获取系统状态
    """
    try:
        # CPU 和内存
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()

        # GPU
        gpus = GPUtil.getGPUs()
        gpu_usage = gpus[0].load * 100 if gpus else 0.0
        gpu_memory = gpus[0].memoryUsed if gpus else None

        return SystemStatus(
            cpu_usage=cpu_usage,
            memory_usage=memory.percent,
            gpu_usage=gpu_usage,
            gpu_memory=gpu_memory,
            current_model=model_manager.current_model_name,
            available_models=model_manager.get_available_models()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取系统状态失败: {str(e)}")


@router.post("/model/switch")
async def switch_model(model_name: str):
    """
    切换模型
    """
    try:
        message = model_manager.switch_model(model_name)
        return {"message": message, "current_model": model_name}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"切换模型失败: {str(e)}")


@router.get("/models")
async def get_available_models():
    """
    获取可用模型列表
    """
    return {
        "models": model_manager.get_available_models(),
        "current": model_manager.current_model_name
    }
