"""
导出服务：PDF 检测报告 + Excel 记录导出
"""
import os
from io import BytesIO
from datetime import datetime
from typing import List

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.lib.utils import ImageReader
from openpyxl import Workbook

from core.config import settings, BACKEND_DIR
from models.db_models import DetectionRecord

# 注册 reportlab 内置 CJK 字体（无需外部字体文件）
_FONT = "STSong-Light"
try:
    pdfmetrics.registerFont(UnicodeCIDFont(_FONT))
except Exception:
    _FONT = "Helvetica"


class ExportService:
    """报告与数据导出"""

    @staticmethod
    def generate_pdf(record: DetectionRecord) -> bytes:
        """为单条检测记录生成 PDF 报告"""
        buffer = BytesIO()
        c = canvas.Canvas(buffer, pagesize=A4)
        width, height = A4

        # 标题
        c.setFont(_FONT, 18)
        c.drawString(25 * mm, height - 30 * mm, "道路病害检测报告")
        c.setStrokeColor(colors.HexColor("#409eff"))
        c.setLineWidth(1.5)
        c.line(25 * mm, height - 33 * mm, width - 25 * mm, height - 33 * mm)

        # 基本信息
        c.setFont(_FONT, 11)
        y = height - 45 * mm
        info_lines = [
            f"记录编号: {record.id}",
            f"检测类型: {record.detection_type}",
            f"文件名称: {record.source_name}",
            f"检测模型: {record.model_name}",
            f"检测时间: {record.created_at.strftime('%Y-%m-%d %H:%M:%S') if record.created_at else '-'}",
            f"置信度阈值: {record.conf_threshold}   IOU阈值: {record.iou_threshold}",
            f"病害总数: {record.total_damages}",
            f"严重程度: {record.severity}   严重指数: {record.severity_score}",
            f"处理耗时: {record.processing_time:.2f}s",
        ]
        for line in info_lines:
            c.drawString(25 * mm, y, line)
            y -= 8 * mm

        # 病害明细
        y -= 4 * mm
        c.setFont(_FONT, 13)
        c.drawString(25 * mm, y, "病害类型明细")
        y -= 8 * mm
        c.setFont(_FONT, 11)
        if record.class_counts:
            for name, cnt in record.class_counts.items():
                c.drawString(30 * mm, y, f"• {name}: {cnt} 处")
                y -= 7 * mm
        else:
            c.drawString(30 * mm, y, "未检测到病害")
            y -= 7 * mm

        # 结果图（如果是图片检测且文件存在）
        if record.result_image_path and record.result_image_path.endswith((".jpg", ".png")):
            fname = os.path.basename(record.result_image_path)
            fpath = os.path.join(settings.RESULT_DIR, fname)
            if os.path.exists(fpath):
                try:
                    y -= 6 * mm
                    c.setFont(_FONT, 13)
                    c.drawString(25 * mm, y, "检测结果图")
                    y -= 6 * mm
                    img = ImageReader(fpath)
                    iw, ih = img.getSize()
                    disp_w = width - 50 * mm
                    disp_h = disp_w * ih / iw
                    max_h = y - 25 * mm
                    if disp_h > max_h:
                        disp_h = max_h
                        disp_w = disp_h * iw / ih
                    c.drawImage(img, 25 * mm, y - disp_h, width=disp_w, height=disp_h)
                except Exception:
                    pass

        # 页脚
        c.setFont(_FONT, 8)
        c.setFillColor(colors.grey)
        c.drawString(25 * mm, 15 * mm, f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  道路病害检测系统 v{settings.VERSION}")

        c.save()
        buffer.seek(0)
        return buffer.getvalue()

    @staticmethod
    def generate_excel(records: List[DetectionRecord]) -> bytes:
        """将检测记录列表导出为 Excel"""
        wb = Workbook()
        ws = wb.active
        ws.title = "检测记录"

        headers = [
            "ID", "检测类型", "文件名", "模型", "置信度阈值", "IOU阈值",
            "病害总数", "严重程度", "严重指数", "病害明细", "处理耗时(s)", "检测时间"
        ]
        ws.append(headers)

        for r in records:
            detail = "; ".join(f"{k}:{v}" for k, v in (r.class_counts or {}).items())
            ws.append([
                r.id,
                r.detection_type,
                r.source_name,
                r.model_name,
                r.conf_threshold,
                r.iou_threshold,
                r.total_damages,
                r.severity,
                r.severity_score,
                detail,
                round(r.processing_time, 2),
                r.created_at.strftime("%Y-%m-%d %H:%M:%S") if r.created_at else "",
            ])

        # 列宽自适应（简单估算）
        widths = [6, 10, 22, 12, 11, 10, 9, 10, 9, 30, 12, 20]
        for i, w in enumerate(widths, 1):
            ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = w

        buffer = BytesIO()
        wb.save(buffer)
        buffer.seek(0)
        return buffer.getvalue()


export_service = ExportService()
