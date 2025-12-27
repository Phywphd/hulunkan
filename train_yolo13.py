#!/usr/bin/env python3
"""
YOLOv13 训练脚本 - 用于与 RT-DETR、YOLO11、YOLOv12 对比
注意: YOLOv13 没有 m 版本，使用 l (Large) 版本
"""
from ultralytics import YOLO
import torch

# 检查GPU
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

# 数据集路径
DATA_YAML = "/home/kemove/ai4s_tmp/ai4sci_work/filtered/data.yaml"

def train():
    # 加载 YOLOv13 预训练模型 (l 版本，因为没有 m)
    model = YOLO('yolov13l.pt')

    # 训练配置 - 与其他模型保持一致
    results = model.train(
        data=DATA_YAML,
        epochs=150,
        imgsz=1280,
        batch=8,  # YOLOv13 也使用 Attention，显存需求大
        device=0,
        workers=8,
        patience=30,
        save=True,
        plots=True,
        project='runs/yolo13',
        name='yolo13l_hulunkan',
        exist_ok=True,

        # 优化器配置
        optimizer='AdamW',
        lr0=0.0001,
        lrf=0.01,
        warmup_epochs=5,

        # Loss 权重
        cls=1.5,
        box=7.5,

        # 数据增强 - 与其他模型一致
        mosaic=1.0,
        mixup=0.2,
        copy_paste=0.3,
        scale=0.9,

        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        flipud=0.0,
        fliplr=0.5,
    )

    print("\n训练完成!")
    print(f"最佳模型: runs/yolo13/yolo13l_hulunkan/weights/best.pt")
    return results


if __name__ == "__main__":
    train()
