#!/usr/bin/env python3
"""
护轮坎人员闯入检测模型训练脚本
"""

from ultralytics import YOLO
from pathlib import Path
import torch

# 检查GPU
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

# 路径配置
PROJECT_DIR = Path(__file__).parent
DATA_YAML = PROJECT_DIR / "data.yaml"

def train_yolov8():
    """训练YOLOv8模型"""
    print("\n" + "=" * 50)
    print("开始训练 YOLOv8m 模型")
    print("=" * 50)

    # 加载预训练模型
    model = YOLO('yolov8m.pt')

    # 训练配置
    # - imgsz=1280: 高分辨率，有利于小目标检测（85%是小目标）
    # - batch=16: 保守值，可根据显存调整
    # - epochs=100: 足够的训练轮数
    # - patience=20: 早停，防止过拟合
    results = model.train(
        data=str(DATA_YAML),
        epochs=100,
        imgsz=1280,
        batch=16,
        device=[0, 1],  # 双GPU训练
        workers=8,
        patience=20,
        save=True,
        plots=True,
        project=str(PROJECT_DIR / "runs"),
        name="yolov8m_hulunkan",
        exist_ok=True,
        # 数据增强
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
    )

    print("\n训练完成！")
    print(f"最佳模型: {PROJECT_DIR}/runs/yolov8m_hulunkan/weights/best.pt")

    return results


def validate_model(model_path):
    """验证模型"""
    print("\n" + "=" * 50)
    print("模型验证")
    print("=" * 50)

    model = YOLO(model_path)
    results = model.val(data=str(DATA_YAML))

    print(f"\nmAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")

    return results


if __name__ == "__main__":
    # 训练模型
    train_yolov8()

    # 验证最佳模型
    best_model = PROJECT_DIR / "runs/yolov8m_hulunkan/weights/best.pt"
    if best_model.exists():
        validate_model(str(best_model))
