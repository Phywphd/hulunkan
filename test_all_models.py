#!/usr/bin/env python3
"""
在 test 集上测试所有模型（RT-DETR、YOLO11、YOLOv12、YOLOv13、RT-DETRv4）
"""
from ultralytics import RTDETR, YOLO
from pathlib import Path
import sys
import subprocess
import os
import re

# 数据集配置
DATA_YAML = "/home/kemove/ai4s_tmp/ai4sci_work/filtered/data.yaml"

# 所有模型路径
MODELS = {
    "rtdetr_v2": {
        "path": "/home/kemove/ai4s_tmp/hulunkan_detection/weights/rtdetr_v2_best.pt",
        "type": "rtdetr",
        "output_dir": "runs/rtdetr/rtdetr_hulunkan_v2_test"
    },
    "yolo11m": {
        "path": "/home/kemove/ai4s_tmp/hulunkan_detection/runs/yolo11/yolo11m_hulunkan/weights/best.pt",
        "type": "yolo",
        "output_dir": "runs/yolo11/yolo11m_hulunkan_test"
    },
    "yolo12m": {
        "path": "/home/kemove/ai4s_tmp/hulunkan_detection/runs/yolo12/yolo12m_hulunkan/weights/best.pt",
        "type": "yolo",
        "output_dir": "runs/yolo12/yolo12m_hulunkan_test"
    },
    "yolo13l": {
        "path": "/home/kemove/ai4s_tmp/hulunkan_detection/runs/yolo13/yolo13l_hulunkan/weights/best.pt",
        "type": "yolo",
        "output_dir": "runs/yolo13/yolo13l_hulunkan_test"
    },
    "rtdetr_v4": {
        "path": "/home/kemove/ai4s_tmp/rtdetrv4_repo/outputs/rtv4_hgnetv2_l_hulunkan/best_stg2.pth",
        "type": "rtdetrv4",
        "config": "/home/kemove/ai4s_tmp/rtdetrv4_repo/configs/rtv4/rtv4_hgnetv2_l_hulunkan.yml",
        "repo_path": "/home/kemove/ai4s_tmp/rtdetrv4_repo",
        "output_dir": "runs/rtdetrv4/rtdetrv4_hulunkan_test"
    },
}


def test_rtdetrv4(config):
    """测试 RT-DETRv4 模型"""
    model_path = config["path"]
    repo_path = config["repo_path"]
    cfg_path = config["config"]

    print("=" * 60)
    print(f"测试模型: RT-DETRv4")
    print(f"模型路径: {model_path}")
    print(f"配置文件: {cfg_path}")
    print("=" * 60)

    # 使用 torchrun 运行测试
    cmd = [
        "torchrun",
        "--nproc_per_node=1",
        "--master_port=29501",
        "train.py",
        "-c", cfg_path,
        "-r", model_path,
        "--test-only",
        "-u", "val_dataloader.dataset.img_folder=/home/kemove/ai4s_tmp/hulunkan_detection/coco_dataset/test",
        "val_dataloader.dataset.ann_file=/home/kemove/ai4s_tmp/hulunkan_detection/coco_dataset/annotations/instances_test.json",
    ]

    print(f"运行命令: {' '.join(cmd)}")

    # 运行命令并捕获输出
    result = subprocess.run(
        cmd,
        cwd=repo_path,
        capture_output=True,
        text=True
    )

    output = result.stdout + result.stderr
    print(output)

    # 解析 COCO 评估结果
    results = parse_coco_results(output)
    return results


def parse_coco_results(output):
    """解析 COCO 评估输出"""
    results = {
        "mAP50": 0.0,
        "mAP50-95": 0.0,
        "mAP75": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "per_class": [0.0, 0.0, 0.0, 0.0]  # person, net, equipment, wheel_guard
    }

    # 匹配 mAP 结果
    map_pattern = r"Average Precision  \(AP\) @\[ IoU=0\.50:0\.95 \| area=   all \| maxDets=100 \] = ([\d.]+)"
    map50_pattern = r"Average Precision  \(AP\) @\[ IoU=0\.50      \| area=   all \| maxDets=100 \] = ([\d.]+)"
    map75_pattern = r"Average Precision  \(AP\) @\[ IoU=0\.75      \| area=   all \| maxDets=100 \] = ([\d.]+)"
    recall_pattern = r"Average Recall     \(AR\) @\[ IoU=0\.50:0\.95 \| area=   all \| maxDets=100 \] = ([\d.]+)"

    map_match = re.search(map_pattern, output)
    map50_match = re.search(map50_pattern, output)
    map75_match = re.search(map75_pattern, output)
    recall_match = re.search(recall_pattern, output)

    if map_match:
        results["mAP50-95"] = float(map_match.group(1))
    if map50_match:
        results["mAP50"] = float(map50_match.group(1))
    if map75_match:
        results["mAP75"] = float(map75_match.group(1))
    if recall_match:
        results["recall"] = float(recall_match.group(1))

    return results


def test_model(model_name):
    """测试单个模型"""
    if model_name not in MODELS:
        print(f"未知模型: {model_name}")
        print(f"可用模型: {list(MODELS.keys())}")
        return None

    config = MODELS[model_name]
    model_type = config["type"]

    # RT-DETRv4 使用单独的测试函数
    if model_type == "rtdetrv4":
        return test_rtdetrv4(config)

    model_path = config["path"]
    output_dir = config["output_dir"]

    print("=" * 60)
    print(f"测试模型: {model_name}")
    print(f"模型路径: {model_path}")
    print(f"输出目录: {output_dir}")
    print("=" * 60)

    # 加载模型
    if model_type == "rtdetr":
        model = RTDETR(model_path)
    else:
        model = YOLO(model_path)

    # 在测试集上评估
    results = model.val(
        data=DATA_YAML,
        split='test',
        project=str(Path(output_dir).parent),
        name=Path(output_dir).name,
        exist_ok=True,
    )

    # 打印结果
    print(f"\n{model_name} 测试集结果:")
    print(f"  Precision: {results.box.mp:.4f}")
    print(f"  Recall:    {results.box.mr:.4f}")
    print(f"  mAP50:     {results.box.map50:.4f}")
    print(f"  mAP50-95:  {results.box.map:.4f}")

    # 打印每个类别的结果
    print("\n  各类别 mAP50-95:")
    class_names = ['person', 'net', 'equipment', 'wheel_guard']
    for i, name in enumerate(class_names):
        if i < len(results.box.maps):
            print(f"    {name}: {results.box.maps[i]:.4f}")

    return results


def test_all():
    """测试所有模型"""
    all_results = {}

    for model_name in MODELS:
        print(f"\n{'#' * 60}")
        print(f"# 开始测试: {model_name}")
        print(f"{'#' * 60}\n")

        results = test_model(model_name)
        if results:
            # RT-DETRv4 返回字典格式
            if isinstance(results, dict):
                all_results[model_name] = results
            else:
                # ultralytics 模型返回结果对象
                all_results[model_name] = {
                    "mAP50": results.box.map50,
                    "mAP50-95": results.box.map,
                    "precision": results.box.mp,
                    "recall": results.box.mr,
                    "per_class": results.box.maps.tolist() if hasattr(results.box.maps, 'tolist') else list(results.box.maps)
                }

    # 打印汇总
    print("\n" + "=" * 80)
    print("所有模型测试集结果汇总")
    print("=" * 80)
    header = f"{'模型':<15} {'mAP50':<10} {'mAP50-95':<12} {'person':<10} {'net':<10} {'equipment':<12} {'wheel_guard':<12}"
    print(header)
    print("-" * 80)

    lines = [header, "-" * 80]
    for name, res in all_results.items():
        per_class = res["per_class"]
        line = f"{name:<15} {res['mAP50']:<10.4f} {res['mAP50-95']:<12.4f} {per_class[0]:<10.4f} {per_class[1]:<10.4f} {per_class[2]:<12.4f} {per_class[3]:<12.4f}"
        print(line)
        lines.append(line)

    # 保存汇总结果到 test_results 目录
    output_dir = Path("/home/kemove/ai4s_tmp/hulunkan_detection/test_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存文本格式
    txt_path = output_dir / "all_models_test_results.txt"
    with open(txt_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("所有模型测试集结果汇总\n")
        f.write("=" * 80 + "\n")
        for line in lines:
            f.write(line + "\n")

    # 保存 CSV 格式
    csv_path = output_dir / "all_models_test_results.csv"
    with open(csv_path, 'w') as f:
        f.write("model,mAP50,mAP50-95,person,net,equipment,wheel_guard\n")
        for name, res in all_results.items():
            per_class = res["per_class"]
            f.write(f"{name},{res['mAP50']:.4f},{res['mAP50-95']:.4f},{per_class[0]:.4f},{per_class[1]:.4f},{per_class[2]:.4f},{per_class[3]:.4f}\n")

    print(f"\n结果已保存到:")
    print(f"  {txt_path}")
    print(f"  {csv_path}")

    return all_results


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # 测试指定模型
        model_name = sys.argv[1]
        test_model(model_name)
    else:
        # 测试所有模型
        test_all()
