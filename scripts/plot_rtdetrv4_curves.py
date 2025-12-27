#!/usr/bin/env python3
"""
解析RT-DETRv4的训练日志并生成训练曲线
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def parse_log(log_path):
    """解析log.txt文件"""
    epochs = []
    train_loss = []
    train_loss_bbox = []
    train_loss_giou = []
    train_loss_distill = []
    lr = []

    # COCO评估指标
    map50_95 = []
    map50 = []

    with open(log_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if 'epoch' in data:
                    epochs.append(data['epoch'])
                    train_loss.append(data.get('train_loss', 0))
                    train_loss_bbox.append(data.get('train_loss_bbox', 0))
                    train_loss_giou.append(data.get('train_loss_giou', 0))
                    train_loss_distill.append(data.get('train_loss_distill', 0))
                    lr.append(data.get('train_lr', 0))

                    # COCO评估指标 [mAP50-95, mAP50, ...]
                    coco_eval = data.get('test_coco_eval_bbox', [0, 0])
                    if len(coco_eval) >= 2:
                        map50_95.append(coco_eval[0])
                        map50.append(coco_eval[1])
            except json.JSONDecodeError:
                continue

    return {
        'epochs': np.array(epochs),
        'train_loss': np.array(train_loss),
        'train_loss_bbox': np.array(train_loss_bbox),
        'train_loss_giou': np.array(train_loss_giou),
        'train_loss_distill': np.array(train_loss_distill),
        'lr': np.array(lr),
        'mAP50-95': np.array(map50_95),
        'mAP50': np.array(map50),
    }

def plot_training_curves(data, output_path):
    """绘制训练曲线 - 只保留关键指标"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # 1. 总损失
    ax = axes[0]
    ax.plot(data['epochs'], data['train_loss'], 'b-', linewidth=1)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('train/total_loss')
    ax.grid(True, alpha=0.3)

    # 2. mAP50-95
    ax = axes[1]
    ax.plot(data['epochs'], data['mAP50-95'], 'b-', linewidth=1)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('mAP')
    ax.set_title('metrics/mAP50-95')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"训练曲线已保存到: {output_path}")

def main():
    log_path = Path("/home/kemove/ai4s_tmp/rtdetrv4_repo/outputs/rtv4_hgnetv2_l_hulunkan/log.txt")
    output_path = Path("/home/kemove/ai4s_tmp/hulunkan_detection/report/figures/results_rtdetrv4.png")

    print(f"解析日志: {log_path}")
    data = parse_log(log_path)

    print(f"共解析 {len(data['epochs'])} 个epoch的数据")
    print(f"最终 mAP50-95: {data['mAP50-95'][-1]:.4f}")
    print(f"最终 mAP50: {data['mAP50'][-1]:.4f}")

    plot_training_curves(data, output_path)

if __name__ == "__main__":
    main()
