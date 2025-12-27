#!/usr/bin/env python3
"""
计算RT-DETRv4每个类别的AP
"""
import json
import torch
import sys
sys.path.insert(0, '/home/kemove/ai4s_tmp/rtdetrv4_repo')

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np

def eval_per_class(ann_file, pred_file):
    """计算每个类别的AP"""
    coco_gt = COCO(ann_file)
    coco_dt = coco_gt.loadRes(pred_file)

    # 类别映射
    cats = coco_gt.loadCats(coco_gt.getCatIds())
    cat_names = {cat['id']: cat['name'] for cat in cats}

    print("=" * 60)
    print("RT-DETRv4-L 各类别检测结果")
    print("=" * 60)
    print(f"{'类别':<15} {'AP@50-95':<12} {'AP@50':<12} {'AP@75':<12}")
    print("-" * 60)

    results = {}
    for cat_id in coco_gt.getCatIds():
        cat_name = cat_names[cat_id]

        # 对单个类别进行评估
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.params.catIds = [cat_id]
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # 获取AP值
        ap50_95 = coco_eval.stats[0]  # AP@[0.5:0.95]
        ap50 = coco_eval.stats[1]     # AP@0.5
        ap75 = coco_eval.stats[2]     # AP@0.75

        results[cat_name] = {
            'AP50-95': ap50_95,
            'AP50': ap50,
            'AP75': ap75
        }

        print(f"{cat_name:<15} {ap50_95:<12.4f} {ap50:<12.4f} {ap75:<12.4f}")

    print("=" * 60)
    return results

def run_inference_and_eval():
    """运行推理并评估"""
    import os
    os.chdir('/home/kemove/ai4s_tmp/rtdetrv4_repo')

    from src.core import YAMLConfig
    from src.solver import TASKS

    # 加载配置
    cfg = YAMLConfig('/home/kemove/ai4s_tmp/rtdetrv4_repo/configs/rtv4/rtv4_hgnetv2_l_hulunkan.yml')

    # 修改为测试集
    cfg.val_dataloader.dataset.img_folder = '/home/kemove/ai4s_tmp/hulunkan_detection/coco_dataset/test'
    cfg.val_dataloader.dataset.ann_file = '/home/kemove/ai4s_tmp/hulunkan_detection/coco_dataset/annotations/instances_test.json'

    # 加载模型
    cfg.resume = '/home/kemove/ai4s_tmp/rtdetrv4_repo/outputs/rtv4_hgnetv2_l_hulunkan/best_stg2.pth'

    solver = TASKS[cfg.yaml_cfg['task']](cfg)
    solver.val()

if __name__ == "__main__":
    # 测试集标注文件
    ann_file = '/home/kemove/ai4s_tmp/hulunkan_detection/coco_dataset/annotations/instances_test.json'

    # 检查是否有预测结果文件
    pred_file = '/home/kemove/ai4s_tmp/rtdetrv4_repo/outputs/rtv4_hgnetv2_l_hulunkan/predictions_test.json'

    import os
    if os.path.exists(pred_file):
        eval_per_class(ann_file, pred_file)
    else:
        print(f"预测文件不存在: {pred_file}")
        print("需要先运行推理生成预测结果")
