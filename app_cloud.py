import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
import time
import base64
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from matplotlib import rcParams
import psutil
import GPUtil
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import hashlib


# ==================== 全局配置 ====================
# 设置中文字体
rcParams["font.family"] = ["SimHei"]
rcParams["axes.unicode_minus"] = False

# 页面配置
st.set_page_config(
    page_title="道路病害检测系统",
    layout="wide",
    page_icon="🛣️",
    initial_sidebar_state="expanded",
)

# 自定义CSS样式
st.markdown(
    """
<style>
    /* 统一卡片样式 */
    .card {
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        background-color: white;
    }
    
    /* 改进选项卡样式 */
    .stTabs [role="tablist"] {
        margin-bottom: 1rem;
    }
    
    .stTabs [role="tab"][aria-selected="true"] {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    
    /* 改进表格样式 */
    .dataframe {
        border-radius: 8px;
    }
    
    /* 改进按钮组 */
    .stButton>button {
        transition: all 0.3s ease;
        border-radius: 5px;
        font-weight: bold;
        padding: 0.5rem;
        margin: 0.25rem 0;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .stDownloadButton>button {
        background-color: #4CAF50;
        color: white;
    }
    
    /* 严重程度提示框 */
    .severity-box {
        padding: 1rem;
        margin: 1rem 0;
        background-color: #f8f9fa;
        border-radius: 5px;
    }
    
    /* 调整列间距 */
    .st-emotion-cache-1v0mbdj {
        padding: 0.5rem;
    }
    
    /* 按钮容器样式 */
    .button-container {
        display: flex;
        gap: 10px;
        margin-bottom: 15px;
    }
    
    /* 压缩包上传样式 */
    .zip-uploader {
        margin-bottom: 20px;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ==================== 模型管理 ====================
MODEL_PATHS = {
    "标准模型": "./pt/CBAM2_DSConv2D_3_UIoU.pt",
    "高精模型": "./pt/CBAM2_DSConv2D_3_UIoU_200epochs.pt",
    "轻量模型": "./pt/DSConv2D_3_aug.pt",
}

CLASS_NAMES = {0: "纵向裂缝", 1: "横向裂缝", 2: "块状裂缝", 3: "坑洼", 4: "修补"}

SEVERITY_LEVELS = {
    "无病害": {"color": "#4CAF50", "max": 0},
    "轻微": {"color": "#8BC34A", "max": 30},
    "中等": {"color": "#FFC107", "max": 50},
    "严重": {"color": "#FF5722", "max": 100},
    "危险": {"color": "#F44336", "max": float("inf")},
}


@st.cache_resource
def load_model(model_name: str = "标准模型"):
    try:
        model = YOLO(MODEL_PATHS[model_name])
        st.sidebar.success(f"模型 '{model_name}' 加载成功!")
        return model
    except Exception as e:
        st.sidebar.error(f"模型加载失败: {e}")
        return None


# ==================== 核心功能函数 ====================
def save_uploaded_file(uploaded_file) -> Optional[str]:
    try:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix="." + uploaded_file.name.split(".")[-1]
        ) as tmp:
            tmp.write(uploaded_file.getvalue())
            return tmp.name
    except Exception as e:
        st.error(f"文件保存失败: {e}")
        return None


def extract_zip(uploaded_zip) -> List[str]:
    """解压ZIP文件并返回所有图片路径"""
    temp_dir = tempfile.mkdtemp()
    image_paths = []

    try:
        with zipfile.ZipFile(uploaded_zip, "r") as zip_ref:
            for file in zip_ref.namelist():
                if file.lower().endswith((".png", ".jpg", ".jpeg")):
                    zip_ref.extract(file, temp_dir)
                    image_paths.append(os.path.join(temp_dir, file))
        return image_paths
    except Exception as e:
        st.error(f"解压ZIP文件失败: {e}")
        return []


def analyze_results(results, selected_classes) -> Dict[str, int]:
    class_counts = {}
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls.item())
            if CLASS_NAMES[cls_id] in selected_classes:
                class_counts[CLASS_NAMES[cls_id]] = (
                    class_counts.get(CLASS_NAMES[cls_id], 0) + 1
                )
    return class_counts


def assess_severity(class_counts: Dict[str, int]) -> Tuple[str, str]:
    total = sum(class_counts.values())
    if total == 0:
        return "无病害", SEVERITY_LEVELS["无病害"]["color"]

    for level, criteria in SEVERITY_LEVELS.items():
        if total <= criteria["max"]:
            return level, criteria["color"]

    return "危险", SEVERITY_LEVELS["危险"]["color"]


def plot_stats(class_counts: Dict[str, int], save_path: Optional[str] = None):
    if not class_counts:
        st.warning("未检测到病害")
        return

    severity, color = assess_severity(class_counts)

    df = pd.DataFrame(list(class_counts.items()), columns=["类别", "数量"])
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(data=df, x="类别", y="数量", palette="viridis")

    for p in ax.patches:
        ax.annotate(
            f"{int(p.get_height())}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            xytext=(0, 9),
            textcoords="offset points",
        )

    plt.title(f"病害统计 - 严重程度: {severity}", fontsize=14, color=color)
    plt.xticks(rotation=15)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    st.pyplot(plt.gcf())
    plt.close()


def get_system_stats() -> Dict[str, float]:
    cpu_percent = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    gpus = GPUtil.getGPUs()
    gpu_load = gpus[0].load * 100 if gpus else 0

    return {
        "CPU使用率": cpu_percent,
        "内存使用率": memory.percent,
        "GPU使用率": gpu_load,
    }


def generate_pdf_report(data: dict, stat_img_path: str) -> bytes:
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    try:
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont

        pdfmetrics.registerFont(TTFont("SimHei", "SimHei.ttf"))
        c.setFont("SimHei", 16)
    except:
        c.setFont("Helvetica", 16)

    c.drawString(72, height - 72, "道路病害检测报告")
    c.line(72, height - 80, width - 72, height - 80)

    try:
        c.setFont("SimHei", 12)
    except:
        c.setFont("Helvetica", 12)

    y_position = height - 100
    c.drawString(72, y_position, f"检测时间: {data['time']}")
    y_position -= 20
    c.drawString(72, y_position, f"检测类型: {data['type']}")
    y_position -= 20
    c.drawString(72, y_position, f"严重程度: {data['severity']}")
    y_position -= 20
    c.drawString(72, y_position, f"检测到病害总数: {sum(data['counts'].values())}")

    y_position -= 30
    c.drawString(72, y_position, "病害统计:")
    y_position -= 20

    for i, (damage_type, count) in enumerate(data["counts"].items()):
        c.drawString(
            72 + (i % 3) * 180, y_position - (i // 3) * 20, f"{damage_type}: {count}"
        )

    # 添加病害统计图
    if stat_img_path and os.path.exists(stat_img_path):
        y_position -= (len(data["counts"]) // 3 + 2) * 20
        try:
            img_reader = ImageReader(stat_img_path)
            img_width, img_height = Image.open(stat_img_path).size
            aspect = img_height / float(img_width)
            display_width = width - 144
            display_height = display_width * aspect

            if y_position - display_height < 72:
                c.showPage()
                y_position = height - 72

            c.drawImage(
                img_reader,
                72,
                y_position - display_height,
                width=display_width,
                height=display_height,
            )
        except Exception as e:
            pass

    c.save()
    buffer.seek(0)
    return buffer.getvalue()


# ==================== 用户界面 ====================
def initialize_session_state():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "user_settings" not in st.session_state:
        st.session_state.user_settings = {"theme": "light", "alert_threshold": "中等"}
    if "realtime_results" not in st.session_state:
        st.session_state.realtime_results = None


def login():
    st.sidebar.subheader("用户登录")
    username = st.sidebar.text_input("用户名")
    password = st.sidebar.text_input("密码", type="password")

    if st.sidebar.button("登录", key="login_btn"):
        if username == "admin" and password == "admin123":
            st.session_state.authenticated = True
            st.session_state.username = username
            st.sidebar.success("登录成功!")
            st.rerun()
        else:
            st.sidebar.error("用户名或密码错误")


def show_detection_results(results):
    """统一显示检测结果"""
    severity, color = assess_severity(results["counts"])

    with st.container():
        st.markdown("### 检测结果")

        # 严重程度卡片
        st.markdown(
            f"""
        <div class="card" style="border-left: 5px solid {color};">
            <h3 style="color: {color};">整体严重程度: {severity}</h3>
            <p>一共检测到 {sum(results['counts'].values())} 处病害</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # 统计图表卡片
        with st.expander("详细统计", expanded=True):
            plot_stats(results["counts"])


# ==================== 主程序 ====================
def main():
    initialize_session_state()

    # 侧边栏
    with st.sidebar:
        if not st.session_state.authenticated:
            login()
        else:
            st.success(f"欢迎, {st.session_state.username}!")
            if st.button("注销", key="logout_btn"):
                st.session_state.authenticated = False
                st.rerun()

            st.title("⚙️ 系统配置")
            st.header("模型设置")
            model_name = st.selectbox("选择检测模型", list(MODEL_PATHS.keys()))
            model = load_model(model_name)

            st.header("检测参数")
            conf_thresh = st.slider("置信度阈值", 0.1, 1.0, 0.5, step=0.05)
            iou_thresh = st.slider("IOU阈值", 0.1, 1.0, 0.7, step=0.05)
            selected_classes = st.multiselect(
                "选择检测类别",
                list(CLASS_NAMES.values()),
                default=list(CLASS_NAMES.values()),
            )

            st.header("警报设置")
            alert_threshold = st.selectbox(
                "警报阈值", list(SEVERITY_LEVELS.keys())[1:], index=1
            )

            st.header("视频优化设置")
            downscale_factor = st.slider("分辨率缩放比例", 0.1, 1.0, 0.5, step=0.1)
            frame_skip = st.slider("跳帧数", 0, 10, 2)

            # st.header("用户设置")
            # theme = st.selectbox(
            #     "主题",
            #     ["浅色", "深色"],
            #     index=0 if st.session_state.user_settings["theme"] == "light" else 1,
            # )
            # st.session_state.user_settings["theme"] = (
            #     "light" if theme == "浅色" else "dark"
            # )

            st.header("系统状态")
            if st.button("刷新状态", key="refresh_status"):
                system_stats = get_system_stats()
                for k, v in system_stats.items():
                    st.metric(label=k, value=f"{v:.1f}%")

    # 主界面
    if not st.session_state.authenticated:
        st.warning("请先登录以使用系统功能")
        st.stop()

    st.title("🛣️ 智能道路病害检测系统")
    st.markdown(
        """
        <div style="text-align: left; font-size: 1.2rem; margin-bottom: 1.5rem;">
            基于YOLO的道路病害检测与评估系统
        </div>
    """,
        unsafe_allow_html=True,
    )

    # 使用选项卡组织不同功能
    tab1, tab2, tab3 = st.tabs(["📷 图像检测", "🎥 视频检测", "📹 实时检测"])

    # 图像检测标签页
    with tab1:
        st.subheader("图像检测")

        # 上传选项
        upload_option = st.radio(
            "上传方式", ["单张/多张图片", "ZIP压缩包"], horizontal=True
        )

        if upload_option == "单张/多张图片":
            uploaded_files = st.file_uploader(
                "上传道路图像(支持多选)",
                type=["jpg", "png", "jpeg"],
                accept_multiple_files=True,
                key="image_uploader",
            )
        else:
            uploaded_zip = st.file_uploader(
                "上传包含道路图像的ZIP压缩包", type=["zip"], key="zip_uploader"
            )
            if uploaded_zip:
                st.success(f"已上传ZIP文件: {uploaded_zip.name}")
                uploaded_files = []

        # 保存检测结果到session_state
        if "image_results" not in st.session_state:
            st.session_state.image_results = None

        if (upload_option == "单张/多张图片" and uploaded_files) or (
            upload_option == "ZIP压缩包" and uploaded_zip
        ):
            if st.button("开始检测", key="batch_detect"):
                with st.spinner("正在处理图片..."):
                    all_results = []
                    total_counts = {}
                    processed_files = []

                    # 处理ZIP文件
                    if upload_option == "ZIP压缩包":
                        image_paths = extract_zip(uploaded_zip)
                        if not image_paths:
                            st.error("ZIP文件中未找到有效图片")
                            return

                        for img_path in image_paths:
                            try:
                                img_cv = cv2.imread(img_path)
                                if img_cv is not None:
                                    result = model.predict(
                                        img_cv, conf=conf_thresh, iou=iou_thresh
                                    )
                                    counts = analyze_results(result, selected_classes)

                                    # 更新总统计
                                    for k, v in counts.items():
                                        total_counts[k] = total_counts.get(k, 0) + v

                                    # 保存结果
                                    all_results.append(
                                        {
                                            "name": os.path.basename(img_path),
                                            "image": img_cv,
                                            "result": result,
                                            "counts": counts,
                                        }
                                    )
                                    processed_files.append(img_path)
                            except Exception as e:
                                st.error(
                                    f"图片 {os.path.basename(img_path)} 处理失败: {e}"
                                )

                    # 处理单张/多张图片
                    else:
                        for i, img_file in enumerate(uploaded_files):
                            img_path = save_uploaded_file(img_file)
                            if img_path:
                                try:
                                    img_cv = cv2.imread(img_path)
                                    result = model.predict(
                                        img_cv, conf=conf_thresh, iou=iou_thresh
                                    )
                                    counts = analyze_results(result, selected_classes)

                                    # 更新总统计
                                    for k, v in counts.items():
                                        total_counts[k] = total_counts.get(k, 0) + v

                                    # 保存结果
                                    all_results.append(
                                        {
                                            "name": img_file.name,
                                            "image": img_cv,
                                            "result": result,
                                            "counts": counts,
                                        }
                                    )
                                    os.unlink(img_path)
                                    processed_files.append(img_file.name)
                                except Exception as e:
                                    st.error(f"图片 {img_file.name} 处理失败: {e}")
                                    if os.path.exists(img_path):
                                        os.unlink(img_path)

                    if all_results:
                        st.success(f"成功处理 {len(all_results)} 张图片")

                        # 保存结果到session_state
                        st.session_state.image_results = {
                            "all_results": all_results,
                            "total_counts": total_counts,
                            "processed_files": processed_files,
                        }

        # 显示图像检测结果
        if st.session_state.image_results:
            results = st.session_state.image_results
            total_counts = results["total_counts"]
            all_results = results["all_results"]

            # 显示总体结果
            show_detection_results({"counts": total_counts})

            # 显示每张图片的结果
            for i, result in enumerate(all_results):
                with st.expander(f"图片 {i+1}: {result['name']}", expanded=(i == 0)):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(
                            result["image"],
                            use_container_width=True,
                            caption="原始图像",
                        )
                    with col2:
                        res_plotted = result["result"][0].plot()
                        st.image(
                            cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB),
                            use_container_width=True,
                            caption="检测结果",
                        )

                    severity, color = assess_severity(result["counts"])
                    st.markdown(
                        f"""<div class="severity-box" style="border-left: 4px solid {color};">
                            <h4 style="color: {color};">严重程度: {severity}</h4>
                            <p>检测到 {sum(result['counts'].values())} 处病害</p>
                        </div>""",
                        unsafe_allow_html=True,
                    )

            # 生成PDF报告
            st.subheader("导出结果")
            stat_img_path = os.path.join(tempfile.gettempdir(), "image_stats.png")
            # plot_stats(total_counts, stat_img_path)

            pdf_report = generate_pdf_report(
                {
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "type": "图像检测",
                    "counts": total_counts,
                    "severity": assess_severity(total_counts)[0],
                },
                stat_img_path,
            )
            st.download_button(
                "📄 下载PDF报告",
                data=pdf_report,
                file_name="image_damage_report.pdf",
                mime="application/pdf",
            )

    # 视频检测标签页
    with tab2:
        st.subheader("视频检测")
        uploaded_file = st.file_uploader(
            "上传道路视频", type=["mp4", "mov", "avi"], key="video_uploader"
        )

        # 初始化session_state
        if "video_results" not in st.session_state:
            st.session_state.video_results = None

        if uploaded_file is not None:
            # 不显示原视频，直接开始分析
            if st.button("开始视频分析", key="start_video_analysis"):
                video_path = save_uploaded_file(uploaded_file)
                if video_path:
                    cap = cv2.VideoCapture(video_path)
                    stframe = st.empty()
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    duration = total_frames / fps if fps > 0 else 0

                    processed_frames = 0
                    frame_count = 0
                    start_time = time.time()

                    stat_path = os.path.join(tempfile.gettempdir(), "video_stats.png")
                    total_counts = {}
                    max_severity = "无病害"

                    # 显示视频信息
                    info_col1, info_col2, info_col3 = st.columns(3)
                    info_col1.metric("总帧数", total_frames)
                    info_col2.metric("FPS", f"{fps:.1f}")
                    info_col3.metric("时长", f"{duration:.1f}秒")

                    status_text.text(f"正在处理视频... 0/{total_frames} 帧 (0.0%)")

                    # 创建结果容器
                    result_container = st.container()

                    # 创建临时视频文件用于保存结果
                    temp_video_path = os.path.join(
                        tempfile.gettempdir(), "result_video.mp4"
                    )
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    out = cv2.VideoWriter(
                        temp_video_path,
                        fourcc,
                        fps,
                        (
                            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                        ),
                    )

                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break

                        frame_count += 1
                        if frame_count % (frame_skip + 1) != 0:
                            continue

                        # 处理帧
                        h, w = frame.shape[:2]
                        small_frame = cv2.resize(
                            frame,
                            (int(w * downscale_factor), int(h * downscale_factor)),
                        )
                        results = model.predict(
                            small_frame, conf=conf_thresh, iou=iou_thresh
                        )
                        res_plotted = results[0].plot()

                        if downscale_factor != 1.0:
                            res_plotted = cv2.resize(res_plotted, (w, h))

                        # 写入结果视频
                        out.write(res_plotted)

                        counts = analyze_results(results, selected_classes)
                        for k, v in counts.items():
                            total_counts[k] = total_counts.get(k, 0) + 1

                        # 更新最大严重程度
                        severity, _ = assess_severity(counts)
                        if list(SEVERITY_LEVELS.keys()).index(severity) > list(
                            SEVERITY_LEVELS.keys()
                        ).index(max_severity):
                            max_severity = severity

                        # 显示处理进度
                        stframe.image(
                            cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB),
                            use_container_width=True,
                        )
                        processed_frames += 1
                        progress = min(
                            processed_frames / (total_frames // (frame_skip + 1)), 1.0
                        )
                        progress_bar.progress(progress)

                        # 更新状态
                        elapsed_time = time.time() - start_time
                        remaining_time = (elapsed_time / processed_frames) * (
                            (total_frames // (frame_skip + 1)) - processed_frames
                        )
                        status_text.text(
                            f"处理中... {processed_frames}/{total_frames // (frame_skip + 1)} 帧 "
                            f"({progress*100:.1f}%) | 已用时间: {elapsed_time:.1f}s | "
                            f"剩余时间: {remaining_time:.1f}s"
                        )

                    cap.release()
                    out.release()
                    processing_time = time.time() - start_time
                    status_text.success(f"处理完成! 耗时: {processing_time:.1f}秒")

                    # 保存到session_state
                    severity, color = assess_severity(total_counts)
                    st.session_state.video_results = {
                        "total_counts": total_counts,
                        "severity": severity,
                        "max_severity": max_severity,
                        "processed_frames": processed_frames,
                        "duration": duration,
                        "score": min(
                            sum(total_counts.values()) / processed_frames, 1.0
                        ),
                        "video_name": uploaded_file.name,
                        "result_video_path": temp_video_path,
                    }

        # 显示视频检测结果
        if st.session_state.video_results:
            results = st.session_state.video_results
            total_counts = results.get("total_counts", {})
            severity = results.get("severity", "无病害")
            color = SEVERITY_LEVELS.get(severity, {}).get("color", "#6c757d")

            with st.container():
                st.subheader("视频分析结果")

                # 显示检测结果
                show_detection_results(
                    {
                        "counts": total_counts,
                        "severity": severity,
                        "max_severity": results.get("max_severity", "无病害"),
                    }
                )

                # 检查是否需要触发警报
                if (
                    severity
                    in list(SEVERITY_LEVELS.keys())[
                        list(SEVERITY_LEVELS.keys()).index(alert_threshold) :
                    ]
                ):
                    st.error(f"⚠️ 警报: 检测到{severity}级病害!")

                # 导出选项
                st.subheader("导出结果")
                col1, col2 = st.columns(2)

                # 导出结果视频
                if results.get("result_video_path"):
                    with col1:
                        with open(results["result_video_path"], "rb") as f:
                            st.download_button(
                                "🎥 下载检测结果视频",
                                f,
                                file_name="detection_result.mp4",
                                help="下载带有检测结果的视频",
                            )

                # 生成PDF报告
                stat_img_path = os.path.join(tempfile.gettempdir(), "video_stats.png")
                # plot_stats(total_counts, stat_img_path)

                pdf_report = generate_pdf_report(
                    {
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "type": "视频检测",
                        "counts": total_counts,
                        "severity": severity,
                        "max_severity": results.get("max_severity", "无病害"),
                        "video_name": results.get("video_name", ""),
                    },
                    stat_img_path,
                )
                with col2:
                    st.download_button(
                        "📄 下载PDF报告",
                        data=pdf_report,
                        file_name="video_damage_report.pdf",
                        mime="application/pdf",
                    )

    # 实时检测标签页
    with tab3:
        st.subheader("实时摄像头检测")
        st.warning("请确保已连接摄像头并授予访问权限")

        # 摄像头选择
        available_cameras = []
        for i in range(3):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_cameras.append(f"摄像头 {i}")
                cap.release()

        if not available_cameras:
            st.error("未检测到可用摄像头！")
        else:
            selected_cam = st.selectbox("选择摄像头", available_cameras)
            cam_index = int(selected_cam.split()[-1])

            # 初始化实时检测状态
            if "realtime_running" not in st.session_state:
                st.session_state.realtime_running = False
            if "realtime_results" not in st.session_state:
                st.session_state.realtime_results = None
            if "realtime_stats" not in st.session_state:
                st.session_state.realtime_stats = {
                    "total_damages": 0,
                    "total_frames": 0,
                    "start_time": None,
                    "damage_counts": {},
                    "max_severity": "无病害",
                }

            # 实时检测控制
            col1, col2 = st.columns(2)
            with col1:
                if (
                    st.button("启动实时检测", key="start_realtime")
                    and not st.session_state.realtime_running
                ):
                    st.session_state.realtime_running = True
                    st.session_state.realtime_stop = False
                    st.session_state.realtime_stats = {
                        "total_damages": 0,
                        "total_frames": 0,
                        "start_time": time.time(),
                        "damage_counts": {},
                        "max_severity": "无病害",
                    }
                    st.rerun()

            if st.session_state.realtime_running:
                cap = cv2.VideoCapture(cam_index)
                stframe = st.empty()
                stop_button_pressed = st.button("停止检测", key="stop_realtime")

                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                cap.set(cv2.CAP_PROP_FPS, 30)

                stats_container = st.container()
                alert_triggered = False

                while cap.isOpened() and st.session_state.realtime_running:
                    if stop_button_pressed:
                        st.session_state.realtime_running = False
                        st.session_state.realtime_stop = True
                        break

                    ret, frame = cap.read()
                    if not ret:
                        st.error("无法获取摄像头画面")
                        break

                    # 更新总帧数
                    st.session_state.realtime_stats["total_frames"] += 1

                    # 处理帧
                    results = model.predict(
                        frame, conf=conf_thresh, iou=iou_thresh, imgsz=640
                    )
                    res_plotted = results[0].plot()

                    # 分析并更新病害统计
                    counts = analyze_results(results, selected_classes)
                    frame_damages = sum(counts.values())
                    st.session_state.realtime_stats["total_damages"] += frame_damages

                    # 更新各类病害计数
                    for k, v in counts.items():
                        st.session_state.realtime_stats["damage_counts"][k] = (
                            st.session_state.realtime_stats["damage_counts"].get(k, 0)
                            + v
                        )

                    # 更新最大严重程度
                    severity, _ = assess_severity(counts)
                    current_severity_level = list(SEVERITY_LEVELS.keys()).index(
                        severity
                    )
                    max_severity_level = list(SEVERITY_LEVELS.keys()).index(
                        st.session_state.realtime_stats["max_severity"]
                    )
                    if current_severity_level > max_severity_level:
                        st.session_state.realtime_stats["max_severity"] = severity

                    # 检查是否需要触发警报
                    if (
                        not alert_triggered
                        and severity
                        in list(SEVERITY_LEVELS.keys())[
                            list(SEVERITY_LEVELS.keys()).index(alert_threshold) :
                        ]
                    ):
                        st.error(f"⚠️ 警报: 检测到{severity}级病害!")
                        alert_triggered = True

                    # 显示结果
                    stframe.image(
                        cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB),
                        use_container_width=True,
                        caption=f"实时检测 - 帧 {st.session_state.realtime_stats['total_frames']}",
                    )

                    # 更新统计信息
                    with stats_container:
                        elapsed_time = (
                            time.time() - st.session_state.realtime_stats["start_time"]
                        )
                        fps = (
                            st.session_state.realtime_stats["total_frames"]
                            / elapsed_time
                            if elapsed_time > 0
                            else 0
                        )

                        col1, col2, col3 = st.columns(3)
                        col1.metric(
                            "处理帧数", st.session_state.realtime_stats["total_frames"]
                        )
                        col2.metric("FPS", f"{fps:.1f}")
                        col3.metric(
                            "当前帧病害数",
                            frame_damages,
                            delta_color="off",
                            help="当前帧检测到的病害数量",
                        )

                        col4, col5 = st.columns(2)
                        col4.metric(
                            "总病害数",
                            st.session_state.realtime_stats["total_damages"],
                            delta_color="off",
                            help="累计检测到的病害总数",
                        )
                        col5.metric(
                            "最大严重程度",
                            st.session_state.realtime_stats["max_severity"],
                            delta_color="off",
                            help="当前检测到的最高病害严重程度",
                        )

                cap.release()

                # 保存最终结果
                st.session_state.realtime_results = {
                    "counts": st.session_state.realtime_stats["damage_counts"],
                    "max_severity": st.session_state.realtime_stats["max_severity"],
                    "duration": time.time()
                    - st.session_state.realtime_stats["start_time"],
                    "frames": st.session_state.realtime_stats["total_frames"],
                    "total_damages": st.session_state.realtime_stats["total_damages"],
                }
                st.rerun()

        # 显示实时检测结果（在停止检测后）
        if st.session_state.realtime_results and not st.session_state.realtime_running:
            results = st.session_state.realtime_results

            with st.container():
                st.subheader("实时检测结果统计")

                severity, color = assess_severity(results["counts"])
                st.markdown(
                    f"""<div class="card" style="border-left: 5px solid {color};">
                        <h3 style="color: {color};">严重程度: {severity}</h3>
                        <p>检测到 {results['total_damages']} 处病害 (共 {results['frames']} 帧)</p>
                        <p>检测时长: {results['duration']:.1f}秒</p>
                        <p>平均每帧病害数: {results['total_damages']/results['frames']:.2f}</p>
                        <p>最大严重程度: <span style="color: {SEVERITY_LEVELS.get(results['max_severity'], {}).get('color', '#6c757d')}">{results['max_severity']}</span></p>
                    </div>""",
                    unsafe_allow_html=True,
                )

                # 显示统计图表
                # plot_stats(results["counts"])

                # 生成PDF报告
                stat_img_path = os.path.join(
                    tempfile.gettempdir(), "realtime_stats.png"
                )
                plot_stats(results["counts"], stat_img_path)

                pdf_report = generate_pdf_report(
                    {
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "type": "实时检测",
                        "counts": results["counts"],
                        "severity": severity,
                        "max_severity": results["max_severity"],
                        "total_frames": results["frames"],
                        "total_damages": results["total_damages"],
                        "duration": results["duration"],
                    },
                    stat_img_path,
                )
                st.download_button(
                    "📄 下载PDF报告",
                    data=pdf_report,
                    file_name="realtime_damage_report.pdf",
                    mime="application/pdf",
                )


if __name__ == "__main__":
    main()
