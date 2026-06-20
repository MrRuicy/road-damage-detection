# 部署指南

本项目采用**单端口容器化**方案：FastAPI 同时托管前端静态文件与 API，整个应用打包为一个镜像、监听一个端口。既可本地 Docker 运行，也可直接部署到 ModelScope 创空间。

> 💡 开发模式不受影响：本地仍用 `start.bat`（后端 8000 + 前端 5173 热重载）。容器化只在 `frontend/dist` 存在时激活单端口模式，两者互不干扰。

---

## 一、本地 Docker 测试

### 前置
- 安装 Docker Desktop（含 Docker Compose）

### 一键构建运行
```bash
docker compose up --build
```
- 首次构建较慢（需下载 CPU 版 torch、ultralytics 等，约 10~20 分钟）
- 构建完成后访问 **http://localhost:8000**

### 数据持久化
`docker-compose.yml` 已挂载两个卷，容器重启数据不丢：
- `./data/db` → 数据库（检测历史记录）
- `./data/results` → 检测结果图/视频

### 停止
```bash
docker compose down
```

---

## 二、部署到 ModelScope 创空间

创空间支持 **Docker SDK**，可直接运行本项目的 FastAPI + 前端单端口应用。

### 步骤 1：创建创空间
1. 登录 [ModelScope](https://www.modelscope.cn/)，进入「创空间」→「创建创空间」
2. SDK 类型选择 **Docker**
3. 获得一个 git 仓库地址

### 步骤 2：准备 README.md 元信息
创空间通过仓库根目录 `README.md` 顶部的 YAML 头识别配置。**新建或在现有 README 顶部加入**：

```yaml
---
title: 道路病害检测系统
emoji: 🛣️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---
```

关键字段：
- `sdk: docker` —— 声明为 Docker 创空间
- `app_port: 7860` —— 必须与 Dockerfile 的 `EXPOSE` / 应用监听端口一致（本项目已统一为 7860）

### 步骤 3：推送代码
将项目推送到创空间仓库（确保包含以下内容）：

```
Dockerfile          ✅ 必须
.dockerignore       ✅ 必须
backend/            ✅ 含 weights/ 模型权重（约 16MB）
frontend/           ✅ 源码（容器内会自动 npm build）
README.md           ✅ 带上面的 YAML 头
```

**注意**：`dataset/`、`legacy_streamlit/`、`.venv/` 已在 `.dockerignore` 中排除，不会进镜像。但推送 git 时 `dataset/` 也建议排除（已在 `.gitignore`），避免仓库过大。

```bash
git add Dockerfile .dockerignore backend frontend README.md
git commit -m "deploy: 容器化部署道路病害检测系统"
git push
```

### 步骤 4：等待构建
- 创空间自动读取 Dockerfile 并构建（首次较慢）
- 构建成功后，通过分配的 `xxx.modelscope.cn` 域名访问

---

## 三、注意事项

### 数据持久化
创空间免费实例**容器重启后数据会重置**（检测历史清空）。这是平台限制，个人学习场景无妨。如需持久化，需使用创空间的持久化存储功能（如平台提供）。

### 实时检测（摄像头）
浏览器摄像头 API（`getUserMedia`）要求**安全上下文**（HTTPS 或 localhost）。创空间提供 HTTPS 域名，因此实时检测可正常使用；若用自建 HTTP 服务则摄像头不可用。

### CPU 推理性能
镜像使用 **CPU 版 PyTorch**（避免 2GB+ 的 CUDA 依赖）。检测速度：单张图约 0.2~0.5s，视频/实时会慢一些，满足学习演示。如创空间提供 GPU 实例，可改用 CUDA 基础镜像加速。

### 端口一致性
若部署到其他平台需改端口，三处保持一致即可：
- `Dockerfile` 的 `ENV PORT` / `EXPOSE`
- `README.md` 的 `app_port`
- 平台要求的端口

本项目通过环境变量 `PORT` 控制，无需改代码。

---

## 四、本地非 Docker 运行（开发）

容器化改造**不影响**原有开发流程，手动启动前后端即可：

```bash
# 终端 1 - 后端
cd backend
python run.py       # 监听 8000

# 终端 2 - 前端（新终端）
cd frontend
npm run dev         # 监听 5173，热重载
```
开发时访问 http://localhost:5173
