#!/usr/bin/env python3
"""
RT-DETRv4-L 训练脚本 - 用于与其他模型对比
使用官方RT-DETRv4仓库进行训练
"""
import os
import sys
import subprocess
import torch

# RT-DETRv4 仓库路径
RTDETRV4_REPO = "/home/kemove/ai4s_tmp/rtdetrv4_repo"

# 检查GPU
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

def train():
    """运行RT-DETRv4训练"""

    # 配置文件路径
    config_file = "configs/rtv4/rtv4_hgnetv2_l_hulunkan.yml"

    print("=" * 60)
    print("RT-DETRv4-L 护轮坎检测训练")
    print("=" * 60)
    print(f"配置文件: {config_file}")
    print(f"仓库路径: {RTDETRV4_REPO}")
    print("=" * 60)

    # 切换到RT-DETRv4仓库目录
    os.chdir(RTDETRV4_REPO)

    # 构建训练命令 - 使用torchrun启动（即使单GPU也需要）
    cmd = [
        "torchrun",
        "--nproc_per_node=1",
        "--master_port=29500",
        "train.py",
        "-c", config_file,
        "--use-amp",
        "--seed", "0",
    ]

    print(f"训练命令: {' '.join(cmd)}")
    print("=" * 60)

    # 运行训练
    result = subprocess.run(cmd, cwd=RTDETRV4_REPO)

    if result.returncode == 0:
        print("\n训练完成!")
        print(f"模型保存在: {RTDETRV4_REPO}/outputs/rtv4_hgnetv2_l_hulunkan/")
    else:
        print(f"\n训练失败，返回码: {result.returncode}")

    return result.returncode


if __name__ == "__main__":
    train()
