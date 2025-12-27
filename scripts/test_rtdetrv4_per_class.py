#!/usr/bin/env python3
"""
测试RT-DETRv4并计算每个类别的AP
"""
import os
import sys
import json
import torch

# 添加rtdetrv4仓库路径
sys.path.insert(0, '/home/kemove/ai4s_tmp/rtdetrv4_repo')
os.chdir('/home/kemove/ai4s_tmp/rtdetrv4_repo')

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from src.core import YAMLConfig
from src.data import get_coco_api_from_dataset
import src.misc.dist as dist

def main():
    # 初始化分布式环境
    dist.init_distributed()

    # 加载配置
    cfg = YAMLConfig('/home/kemove/ai4s_tmp/rtdetrv4_repo/configs/rtv4/rtv4_hgnetv2_l_hulunkan.yml')

    # 设置测试集路径
    cfg.yaml_cfg['val_dataloader']['dataset']['img_folder'] = '/home/kemove/ai4s_tmp/hulunkan_detection/coco_dataset/test'
    cfg.yaml_cfg['val_dataloader']['dataset']['ann_file'] = '/home/kemove/ai4s_tmp/hulunkan_detection/coco_dataset/annotations/instances_test.json'

    # 加载模型权重
    cfg.resume = '/home/kemove/ai4s_tmp/rtdetrv4_repo/outputs/rtv4_hgnetv2_l_hulunkan/best_stg2.pth'

    # 构建模型
    model = cfg.model
    model.eval()

    # 加载权重
    checkpoint = torch.load(cfg.resume, map_location='cpu')
    if 'ema' in checkpoint and checkpoint['ema'] is not None:
        state = checkpoint['ema']['module']
    else:
        state = checkpoint['model']
    model.load_state_dict(state)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 构建数据加载器
    val_loader = cfg.val_dataloader
    base_ds = get_coco_api_from_dataset(val_loader.dataset)

    # 运行推理
    from src.solver.det_engine import evaluate
    coco_evaluator = evaluate(model, cfg.criterion, cfg.postprocessor, val_loader, base_ds, device)

    # 计算每个类别的AP
    print("\n" + "=" * 70)
    print("RT-DETRv4-L 各类别检测结果 (测试集)")
    print("=" * 70)

    coco_eval = coco_evaluator.coco_eval['bbox']
    coco_gt = coco_eval.cocoGt
    coco_dt = coco_eval.cocoDt

    # 类别映射
    cats = coco_gt.loadCats(coco_gt.getCatIds())
    cat_names = {cat['id']: cat['name'] for cat in cats}

    print(f"{'类别':<15} {'mAP50-95':<12}")
    print("-" * 30)

    per_class_ap = {}
    for cat_id in sorted(coco_gt.getCatIds()):
        cat_name = cat_names[cat_id]

        # 对单个类别进行评估
        eval_single = COCOeval(coco_gt, coco_dt, 'bbox')
        eval_single.params.catIds = [cat_id]
        eval_single.evaluate()
        eval_single.accumulate()

        # 获取AP@[0.5:0.95]
        ap50_95 = eval_single.stats[0]
        per_class_ap[cat_name] = ap50_95

        print(f"{cat_name:<15} {ap50_95:.4f}")

    print("=" * 70)

    # 保存结果
    output_path = '/home/kemove/ai4s_tmp/hulunkan_detection/test_results/rtdetrv4_per_class.json'
    with open(output_path, 'w') as f:
        json.dump(per_class_ap, f, indent=2)
    print(f"\n结果已保存到: {output_path}")

if __name__ == "__main__":
    main()
