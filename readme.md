# road-damage-detection

基于 YOLO + Streamlit 的道路病害检测系统

## 项目简介

通过深度学习目标检测模型，实现对道路图像中各种病害（如裂缝、坑洞等）的自动识别。用户可通过 Web 界面（由 Streamlit 提供）上传图片或视频／摄像头流，系统将返回检测框和类别，辅助道路维护、基础设施巡查等。

## 功能特性

- 使用目标检测模型（比如 YOLO11 权重）识别道路病害。
- 支持通过 Web 界面上传图像进行检测。
- 支持实时视频／摄像头流检测。
- 检测结果以可视化形式输出（标出病害位置、类别）。
- 用户友好界面，操作简单直观。

## 技术栈

- Python
- Streamlit：用于前端 Web 应用
- YOLO11用于目标检测
-  PyTorch 
- 依赖项见 `requirements.txt`

## 安装与使用

### 安装步骤

1. 克隆仓库：

   ```bash
   git clone https://github.com/MrRuicy/road-damage-detection.git  
   cd road-damage-detection  
   ```

2. 创建并激活虚拟环境（推荐）：

   ```bash
   python3 -m venv venv  
   source venv/bin/activate     # Linux/macOS  
   venv\Scripts\activate        # Windows  
   ```

3. 安装依赖：

   ```bash
   pip install -r requirements.txt  
   ```

### 运行应用

```bash
streamlit run app_cloud.py  
```

打开浏览器访问 `http://localhost:8501`（默认端口）即可使用。

## 项目结构

```
road-damage-detection/
├── app_cloud.py             # Streamlit 主程序入口  
├── best.pt                  # 模型权重文件  
├── requirements.txt         # Python 依赖列表  
├── README.md                # 本文档  
```