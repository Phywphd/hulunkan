#!/usr/bin/env python3
"""
为RT-DETRv4生成AP柱状图（替代PR曲线）
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def parse_final_metrics(log_path):
    """解析最后一个epoch的COCO评估指标"""
    with open(log_path, 'r') as f:
        lines = f.readlines()

    # 获取最后一行有效数据
    for line in reversed(lines):
        try:
            data = json.loads(line.strip())
            if 'test_coco_eval_bbox' in data:
                return data['test_coco_eval_bbox']
        except json.JSONDecodeError:
            continue
    return None

def plot_ap_bar(metrics, output_path):
    """绘制AP柱状图"""
    # COCO评估指标含义：
    # [0] AP@[IoU=0.50:0.95]  [1] AP@[IoU=0.50]  [2] AP@[IoU=0.75]
    # [3] AP@[small]  [4] AP@[medium]  [5] AP@[large]
    # [6] AR@[maxDets=1]  [7] AR@[maxDets=10]  [8] AR@[maxDets=100]
    # [9] AR@[small]  [10] AR@[medium]  [11] AR@[large]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    # AP指标
    labels = ['AP\n@50:95', 'AP\n@50', 'AP\n@75', 'AP\n@small', 'AP\n@medium', 'AP\n@large']
    values = metrics[:6]

    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#00BCD4', '#E91E63'],
                  edgecolor='black', linewidth=0.5)

    # 添加数值标签
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Average Precision (AP)')
    ax.set_xlabel('Metric')
    ax.set_title('RT-DETRv4-L COCO Evaluation Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"AP柱状图已保存到: {output_path}")

def main():
    log_path = Path("/home/kemove/ai4s_tmp/rtdetrv4_repo/outputs/rtv4_hgnetv2_l_hulunkan/log.txt")
    output_path = Path("/home/kemove/ai4s_tmp/hulunkan_detection/report/figures/ap_bar_rtdetrv4.png")

    print(f"解析日志: {log_path}")
    metrics = parse_final_metrics(log_path)

    if metrics:
        print(f"AP@50-95: {metrics[0]:.4f}, AP@50: {metrics[1]:.4f}, AP@75: {metrics[2]:.4f}")
        plot_ap_bar(metrics, output_path)
    else:
        print("未找到COCO评估指标")

if __name__ == "__main__":
    main()
