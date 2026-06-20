"""
通用工具函数
"""
import os
import hashlib
from typing import Optional
from datetime import datetime
import cv2
import numpy as np
from fastapi import UploadFile


async def save_upload_file(upload_file: UploadFile, upload_dir: str) -> str:
    """保存上传的文件"""
    os.makedirs(upload_dir, exist_ok=True)

    # 生成唯一文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_ext = os.path.splitext(upload_file.filename)[1]
    filename = f"{timestamp}_{upload_file.filename}"
    file_path = os.path.join(upload_dir, filename)

    # 保存文件
    with open(file_path, "wb") as f:
        content = await upload_file.read()
        f.write(content)

    return file_path


def read_image_file(file_path: str) -> Optional[np.ndarray]:
    """读取图片文件"""
    try:
        image = cv2.imread(file_path)
        if image is None:
            return None
        return image
    except Exception as e:
        print(f"读取图片失败: {e}")
        return None


def compute_file_hash(file_path: str) -> str:
    """计算文件内容的 MD5（用于去重，基于原始字节，分块读取省内存）"""
    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)
    return md5.hexdigest()


def encode_image_to_base64(image: np.ndarray, format: str = ".jpg") -> str:
    """将图片编码为base64"""
    import base64

    _, buffer = cv2.imencode(format, image)
    img_base64 = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{img_base64}"
