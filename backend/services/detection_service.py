"""
检测服务
"""
import time
from typing import Dict, List, Tuple
import cv2
import numpy as np
from ultralytics.engine.results import Results

from core.config import settings
from models.yolo_model import model_manager


class DetectionService:
    """检测服务类"""

    # 严重度等级阈值：基于 0~100 的加权病害指数 severity_score
    SEVERITY_THRESHOLDS = [
        ("无病害", 0.0),
        ("轻微", 20.0),
        ("中等", 45.0),
        ("严重", 70.0),
        ("危险", float("inf")),
    ]

    @staticmethod
    def analyze_results(results: List[Results], selected_classes: List[str] = None) -> Dict[str, int]:
        """分析检测结果，统计各类病害数量"""
        if selected_classes is None:
            selected_classes = list(settings.CLASS_NAMES.values())

        class_counts = {}
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls.item())
                class_name = settings.CLASS_NAMES.get(cls_id, "未知")

                if class_name in selected_classes:
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1

        return class_counts

    @staticmethod
    def compute_severity_score(boxes: List[dict], image_area: float = None) -> float:
        """
        计算加权病害指数 (0~100)。

        综合三个维度：
        - 病害类型权重（坑洼 > 块状裂缝 > 横/纵裂缝 > 修补）
        - 检测置信度（越高越确信病害存在）
        - 检测框面积占比（病害越大越严重，无图像尺寸时忽略此项）

        每个框贡献 = 权重 × 置信度 × (1 + 面积占比放大系数)
        总分经对数压缩映射到 0~100，避免数量线性堆高。
        """
        if not boxes:
            return 0.0

        raw = 0.0
        for b in boxes:
            weight = settings.CLASS_WEIGHTS.get(b["class_name"], 1.0)
            conf = b.get("confidence", 0.5)

            area_factor = 0.0
            if image_area and image_area > 0:
                box_area = max(0.0, (b["x2"] - b["x1"])) * max(0.0, (b["y2"] - b["y1"]))
                ratio = box_area / image_area
                # 面积占比放大：单个框占满全图最多额外 +1.0 倍贡献
                area_factor = min(ratio * 3.0, 1.0)

            raw += weight * conf * (1.0 + area_factor)

        # 对数压缩：raw 越大增长越缓，25 个标准坑洼≈满分附近
        import math
        score = 100.0 * (1.0 - math.exp(-raw / 8.0))
        return round(min(score, 100.0), 1)

    @staticmethod
    def severity_from_score(score: float) -> str:
        """根据加权指数得出严重度等级"""
        if score <= 0:
            return "无病害"
        for level, upper in DetectionService.SEVERITY_THRESHOLDS:
            if level == "无病害":
                continue
            if score <= upper:
                return level
        return "危险"

    @staticmethod
    def assess_severity(class_counts: Dict[str, int]) -> str:
        """
        兼容旧接口：仅凭类别计数估算等级（无置信度/面积信息时使用）。
        新代码请优先用 compute_severity_score + severity_from_score。
        """
        if not class_counts:
            return "无病害"
        # 用权重对计数加权后走同一套对数映射
        pseudo_boxes = []
        for name, cnt in class_counts.items():
            for _ in range(cnt):
                pseudo_boxes.append({
                    "class_name": name,
                    "confidence": 0.7,
                    "x1": 0, "y1": 0, "x2": 0, "y2": 0,
                })
        score = DetectionService.compute_severity_score(pseudo_boxes)
        return DetectionService.severity_from_score(score)

    @staticmethod
    def plot_result(results: List[Results]) -> np.ndarray:
        """生成带检测框的可视化图片 (BGR)"""
        return results[0].plot()

    @staticmethod
    def extract_boxes(results: List[Results]) -> List[dict]:
        """提取检测框信息"""
        boxes = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls.item())
                xyxy = box.xyxy[0].cpu().numpy()

                boxes.append({
                    "x1": float(xyxy[0]),
                    "y1": float(xyxy[1]),
                    "x2": float(xyxy[2]),
                    "y2": float(xyxy[3]),
                    "confidence": float(box.conf.item()),
                    "class_id": cls_id,
                    "class_name": settings.CLASS_NAMES.get(cls_id, "未知"),
                })

        return boxes

    @staticmethod
    async def detect_image(
        image: np.ndarray,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.7,
        model_name: str = None
    ) -> Tuple[List[Results], float]:
        """检测单张图片"""
        start_time = time.time()

        model = model_manager.get_model(model_name)
        results = model.predict(
            image,
            conf=conf_threshold,
            iou=iou_threshold
        )

        processing_time = time.time() - start_time
        return results, processing_time


detection_service = DetectionService()
