"""
检测相关 API 路由
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Depends
from typing import List
import os

from sqlalchemy.ext.asyncio import AsyncSession

from schemas.detection import (
    DetectionResult,
    BatchDetectionResult,
    DetectionBox
)
from services.detection_service import detection_service
from services.record_service import record_service
from utils.file_utils import save_upload_file, read_image_file, encode_image_to_base64, compute_file_hash
from core.config import settings
from core.database import get_db

router = APIRouter()


@router.post("/image", response_model=DetectionResult)
async def detect_image(
    file: UploadFile = File(...),
    conf_threshold: float = Form(0.5),
    iou_threshold: float = Form(0.7),
    model_name: str = Form(settings.DEFAULT_MODEL),
    db: AsyncSession = Depends(get_db),
):
    """
    单张图片检测，结果持久化入库
    """
    # 验证文件类型
    if not any(file.filename.lower().endswith(ext) for ext in settings.ALLOWED_IMAGE_EXTENSIONS):
        raise HTTPException(status_code=400, detail="不支持的图片格式")

    file_path = None
    try:
        # 保存上传文件
        file_path = await save_upload_file(file, settings.UPLOAD_DIR)

        # 读取图片
        image = read_image_file(file_path)
        if image is None:
            raise HTTPException(status_code=400, detail="无法读取图片")

        image_area = float(image.shape[0] * image.shape[1])

        # 智能去重：内容+模型+参数完全一致时，复用已有结果，不重新检测也不新增记录
        content_hash = compute_file_hash(file_path)
        dup = await record_service.find_duplicate(
            db,
            content_hash=content_hash,
            model_name=model_name,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )
        if dup is not None:
            # 清理临时文件
            if os.path.exists(file_path):
                os.remove(file_path)
            # 从磁盘读取旧结果图编码返回
            result_image = None
            if dup.result_image_path:
                fpath = os.path.join(settings.RESULT_DIR, os.path.basename(dup.result_image_path))
                old = read_image_file(fpath)
                if old is not None:
                    result_image = encode_image_to_base64(old)
            return DetectionResult(
                image_name=file.filename,
                boxes=[DetectionBox(**b) for b in (dup.boxes or [])],
                class_counts=dup.class_counts or {},
                total_damages=dup.total_damages,
                severity=dup.severity,
                severity_score=dup.severity_score,
                processing_time=dup.processing_time,
                result_image=result_image,
                record_id=dup.id,
                is_duplicate=True,
            )

        # 执行检测
        results, processing_time = await detection_service.detect_image(
            image, conf_threshold, iou_threshold, model_name
        )

        # 分析结果 + 加权严重度评估
        class_counts = detection_service.analyze_results(results)
        boxes = detection_service.extract_boxes(results)
        severity_score = detection_service.compute_severity_score(boxes, image_area)
        severity = detection_service.severity_from_score(severity_score)

        # 生成带框结果图：base64 即时返回 + 持久化到静态目录
        plotted = detection_service.plot_result(results)
        result_image = encode_image_to_base64(plotted)
        result_image_path = record_service.save_result_image(plotted, prefix="image")

        # 清理临时文件
        if os.path.exists(file_path):
            os.remove(file_path)

        # 入库
        record = await record_service.create_record(
            db,
            detection_type="image",
            source_name=file.filename,
            model_name=model_name,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            total_damages=sum(class_counts.values()),
            class_counts=class_counts,
            severity=severity,
            severity_score=severity_score,
            result_image_path=result_image_path,
            boxes=boxes,
            processing_time=processing_time,
            content_hash=content_hash,
        )

        return DetectionResult(
            image_name=file.filename,
            boxes=[DetectionBox(**box) for box in boxes],
            class_counts=class_counts,
            total_damages=sum(class_counts.values()),
            severity=severity,
            severity_score=severity_score,
            processing_time=processing_time,
            result_image=result_image,
            record_id=record.id,
        )

    except HTTPException:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
        raise
    except Exception as e:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"检测失败: {str(e)}")


@router.post("/batch", response_model=BatchDetectionResult)
async def detect_batch(
    files: List[UploadFile] = File(...),
    conf_threshold: float = Form(0.5),
    iou_threshold: float = Form(0.7),
    model_name: str = Form(settings.DEFAULT_MODEL),
    db: AsyncSession = Depends(get_db),
):
    """
    批量图片检测，每张结果均入库；结果图存磁盘返回 URL（不内联 base64）
    """
    if len(files) > 50:
        raise HTTPException(status_code=400, detail="单次最多上传50张图片")

    all_results = []
    total_class_counts = {}
    all_boxes = []
    file_paths = []

    try:
        for file in files:
            # 验证文件类型
            if not any(file.filename.lower().endswith(ext) for ext in settings.ALLOWED_IMAGE_EXTENSIONS):
                continue

            # 保存并检测
            file_path = await save_upload_file(file, settings.UPLOAD_DIR)
            file_paths.append(file_path)

            image = read_image_file(file_path)
            if image is None:
                continue

            image_area = float(image.shape[0] * image.shape[1])

            results, processing_time = await detection_service.detect_image(
                image, conf_threshold, iou_threshold, model_name
            )

            class_counts = detection_service.analyze_results(results)
            boxes = detection_service.extract_boxes(results)
            severity_score = detection_service.compute_severity_score(boxes, image_area)
            severity = detection_service.severity_from_score(severity_score)

            # 批量模式：结果图存磁盘返回 URL，避免响应体臃肿
            plotted = detection_service.plot_result(results)
            result_image_path = record_service.save_result_image(plotted, prefix="batch")

            # 累计统计
            for k, v in class_counts.items():
                total_class_counts[k] = total_class_counts.get(k, 0) + v
            all_boxes.extend(boxes)

            # 入库
            record = await record_service.create_record(
                db,
                detection_type="batch",
                source_name=file.filename,
                model_name=model_name,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                total_damages=sum(class_counts.values()),
                class_counts=class_counts,
                severity=severity,
                severity_score=severity_score,
                result_image_path=result_image_path,
                boxes=boxes,
                processing_time=processing_time,
            )

            all_results.append(DetectionResult(
                image_name=file.filename,
                boxes=[DetectionBox(**box) for box in boxes],
                class_counts=class_counts,
                total_damages=sum(class_counts.values()),
                severity=severity,
                severity_score=severity_score,
                processing_time=processing_time,
                result_image=result_image_path,  # 这里放 URL
                record_id=record.id,
            ))

        # 清理临时文件
        for fp in file_paths:
            if os.path.exists(fp):
                os.remove(fp)

        overall_score = detection_service.compute_severity_score(all_boxes)
        overall_severity = detection_service.severity_from_score(overall_score)

        return BatchDetectionResult(
            total_images=len(all_results),
            total_damages=sum(total_class_counts.values()),
            overall_severity=overall_severity,
            overall_severity_score=overall_score,
            class_counts=total_class_counts,
            results=all_results
        )

    except Exception as e:
        # 清理所有临时文件
        for fp in file_paths:
            if os.path.exists(fp):
                os.remove(fp)
        raise HTTPException(status_code=500, detail=f"批量检测失败: {str(e)}")
