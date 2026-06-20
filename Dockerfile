# ============================================================
# 道路病害检测系统 - 容器镜像（前端 + 后端单端口）
# 多阶段构建：阶段1 打包前端，阶段2 运行 FastAPI（同时托管前端）
# ============================================================

# ---------- 阶段1：构建前端 ----------
FROM node:20-slim AS frontend-builder
WORKDIR /build

# 先装依赖（利用 Docker 层缓存：package 不变则跳过重装）
COPY frontend/package*.json ./
RUN npm ci

# 构建
COPY frontend/ ./
RUN npm run build
# 产物位于 /build/dist


# ---------- 阶段2：后端运行 ----------
FROM python:3.10-slim

# OpenCV / ultralytics 运行所需系统库
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 关键优化：先装 CPU 版 torch，避免 ultralytics 默认拉取 ~2GB 的 CUDA 版
RUN pip install --no-cache-dir \
        torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu

# 后端 Python 依赖（单独 COPY 以利用层缓存）
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# 后端代码（含 backend/weights/ 模型权重）
COPY backend/ ./backend/

# 前端构建产物（来自阶段1）→ main.py 期望路径 /app/frontend/dist
COPY --from=frontend-builder /build/dist ./frontend/dist

# 运行配置（可被部署平台的环境变量覆盖）
ENV DEBUG=false
ENV HOST=0.0.0.0
ENV PORT=7860
# ultralytics 配置目录指向可写位置（容器内 /root/.config 可能只读）
ENV YOLO_CONFIG_DIR=/tmp/Ultralytics
EXPOSE 7860

# 以 backend 为工作目录启动（匹配 main:app 的相对导入与路径锚定）
# run.py 通过 config 读取 HOST/PORT/DEBUG 环境变量，端口可由平台注入覆盖
WORKDIR /app/backend
CMD ["python", "run.py"]
