"""
YOLO 模型管理器
"""
from typing import Optional
from ultralytics import YOLO
from core.config import settings


class ModelManager:
    def __init__(self):
        self.models = {}
        self.current_model_name = settings.DEFAULT_MODEL
        # 启动时尝试预加载默认模型，失败不影响应用启动
        try:
            self._load_model(self.current_model_name)
        except Exception as e:
            print(f"⚠ 默认模型预加载失败（将在首次使用时重试）: {e}")

    def _load_model(self, model_name: str) -> YOLO:
        """加载指定模型"""
        if model_name not in self.models:
            model_path = settings.MODEL_PATHS.get(model_name)
            if not model_path:
                raise ValueError(f"未知的模型: {model_name}")

            try:
                self.models[model_name] = YOLO(model_path)
                print(f"✓ 模型 '{model_name}' 加载成功")
            except Exception as e:
                raise RuntimeError(f"模型加载失败: {e}")

        return self.models[model_name]

    def get_model(self, model_name: Optional[str] = None) -> YOLO:
        """获取模型实例"""
        if model_name is None:
            model_name = self.current_model_name

        if model_name not in self.models:
            self._load_model(model_name)

        return self.models[model_name]

    def switch_model(self, model_name: str):
        """切换当前模型"""
        if model_name not in settings.MODEL_PATHS:
            raise ValueError(f"未知的模型: {model_name}")

        self._load_model(model_name)
        self.current_model_name = model_name
        return f"已切换到模型: {model_name}"

    def get_available_models(self):
        """获取可用模型列表"""
        return list(settings.MODEL_PATHS.keys())


# 全局模型管理器实例
model_manager = ModelManager()
