"""
生成道路病害检测测试视频

从 dataset 真实街景图合成一段「行车驶近」效果的视频（缓慢平移 + 推近），
用于测试视频检测的追踪去重、严重度评估与在线播放。

用法：
    python scripts/make_test_video.py                 # 默认参数生成
    python scripts/make_test_video.py --out my.mp4 --seconds 12

输出默认为项目根目录 sample_road_damage.mp4
"""
import argparse
import os

import cv2
import numpy as np

# 病害较丰富的街景图（行车视角），按病害数从多到少排列
DEFAULT_IMAGES = [
    "China_MotorBike_001395.jpg",
    "China_MotorBike_000856.jpg",
    "China_MotorBike_000312.jpg",
    "China_MotorBike_001944.jpg",
    "China_MotorBike_000406.jpg",
]

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
IMG_DIR = os.path.join(PROJECT_ROOT, "dataset", "images", "test")


def ken_burns(img, n_frames, out_w, out_h, zoom_end=1.25, pan=0.08):
    """对单张图做缓慢推近 + 横向平移，产出 n_frames 帧（模拟行车运镜）。"""
    h, w = img.shape[:2]
    frames = []
    for i in range(n_frames):
        t = i / max(n_frames - 1, 1)
        zoom = 1.0 + (zoom_end - 1.0) * t          # 1.0 → zoom_end 逐渐推近
        cw, ch = int(w / zoom), int(h / zoom)
        # 横向平移：裁剪窗口从左缓慢移到右
        max_x = w - cw
        x0 = int(max_x * (0.5 - pan + 2 * pan * t))
        x0 = max(0, min(x0, max_x))
        y0 = (h - ch) // 2
        crop = img[y0:y0 + ch, x0:x0 + cw]
        frames.append(cv2.resize(crop, (out_w, out_h)))
    return frames


def main():
    parser = argparse.ArgumentParser(description="生成道路病害检测测试视频")
    parser.add_argument("--out", default=os.path.join(PROJECT_ROOT, "sample_road_damage.mp4"),
                        help="输出视频路径")
    parser.add_argument("--seconds", type=float, default=10.0, help="视频总时长（秒）")
    parser.add_argument("--fps", type=int, default=15, help="帧率")
    parser.add_argument("--width", type=int, default=1280, help="输出宽度")
    parser.add_argument("--height", type=int, default=720, help="输出高度")
    args = parser.parse_args()

    # 收集存在的素材图
    images = []
    for name in DEFAULT_IMAGES:
        p = os.path.join(IMG_DIR, name)
        img = cv2.imread(p)
        if img is not None:
            images.append(img)
    if not images:
        raise SystemExit(f"未找到素材图，请确认 {IMG_DIR} 下存在 China_MotorBike_*.jpg")

    total_frames = int(args.seconds * args.fps)
    per_img = max(total_frames // len(images), 1)

    writer = cv2.VideoWriter(
        args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (args.width, args.height)
    )

    for idx, img in enumerate(images):
        for f in ken_burns(img, per_img, args.width, args.height):
            writer.write(f)
        print(f"  [{idx + 1}/{len(images)}] 已写入 {per_img} 帧")
    writer.release()

    size_kb = os.path.getsize(args.out) / 1024
    print(f"\n✓ 测试视频已生成: {args.out}")
    print(f"  时长 ~{args.seconds:.0f}s | {args.fps}fps | {args.width}x{args.height} | {size_kb:.0f} KB")
    print(f"  含 {len(images)} 个场景，每个场景行车推近运镜，可测试追踪去重")


if __name__ == "__main__":
    main()
