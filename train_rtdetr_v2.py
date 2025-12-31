#!/usr/bin/env python3
"""
RT-DETR 训练脚本 v2
针对小目标 equipment 优化
"""
from ultralytics import RTDETR
import torch

# 检查GPU
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

# 数据集路径 - 使用相对路径
DATA_YAML = "data.yaml"

def train():
    # 加载 RT-DETR 预训练模型
    model = RTDETR('rtdetr-l.pt')

    # 训练配置 - 针对小目标优化
    results = model.train(
        data=DATA_YAML,
        epochs=150,              # 增加训练轮次
        imgsz=1280,
        batch=32,
        device=0,
        workers=8,
        patience=30,             # 增加早停耐心
        save=True,
        plots=True,
        project='runs/rtdetr',
        name='rtdetr_hulunkan_v2',
        exist_ok=True,

        # 优化器配置
        optimizer='AdamW',
        lr0=0.0001,
        lrf=0.01,
        warmup_epochs=5,         # 增加 warmup

        # Loss 权重调整 - 增加分类权重
        cls=1.5,                 # 增加分类 loss 权重 (默认0.5)
        box=7.5,                 # box loss 权重

        # 数据增强 - 针对小目标
        mosaic=1.0,
        mixup=0.2,               # 增加 mixup
        copy_paste=0.3,          # 开启 copy-paste 增强小目标
        scale=0.9,               # 缩放增强

        # 其他增强
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,             # 轻微旋转
        translate=0.1,
        flipud=0.0,
        fliplr=0.5,
    )

    print("\n训练完成!")
    print(f"最佳模型: runs/rtdetr/rtdetr_hulunkan_v2/weights/best.pt")
    return results


if __name__ == "__main__":
    train()
