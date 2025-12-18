#!/usr/bin/env python3
"""
护轮坎人员闯入检测推理脚本

报警逻辑：
- 人员进入护轮坎区域 且 (未穿救生衣 或 未设置安全网) → 报警(1)
- 人员不在护轮坎区域 或 (安全网+救生衣均存在) → 不报警(0)

输出：
- 二分类结果：1=报警, 0=不报警
- 如果报警，输出闯入人员的检测框坐标
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import cv2
import json
from datetime import datetime


# 类别定义
CLASS_NAMES = ['person', 'net', 'equipment']
CLASS_PERSON = 0
CLASS_NET = 1
CLASS_EQUIPMENT = 2

# 默认模型路径
DEFAULT_MODEL = Path(__file__).parent / "runs/yolov8m_hulunkan3/weights/best.pt"

# 默认护轮坎区域 (归一化坐标 0-1)
# 格式: [x1, y1, x2, y2]
# 根据示例图片，护轮坎区域大致在图像左中部
DEFAULT_REGION = [0.1, 0.25, 0.45, 0.85]


def load_model(model_path):
    """加载YOLO模型"""
    model = YOLO(str(model_path))
    print(f"模型已加载: {model_path}")
    return model


def parse_region(region_str):
    """解析区域字符串为坐标列表"""
    if region_str is None:
        return DEFAULT_REGION
    try:
        coords = [float(x.strip()) for x in region_str.split(',')]
        if len(coords) != 4:
            raise ValueError("区域需要4个坐标值")
        return coords
    except Exception as e:
        print(f"区域解析错误: {e}，使用默认区域")
        return DEFAULT_REGION


def box_in_region(bbox, region, img_width, img_height, overlap_threshold=0.3):
    """
    判断检测框是否在护轮坎区域内

    Args:
        bbox: 检测框 [x1, y1, x2, y2] 像素坐标
        region: 护轮坎区域 [x1, y1, x2, y2] 归一化坐标
        img_width, img_height: 图像尺寸
        overlap_threshold: 重叠比例阈值

    Returns:
        bool: 是否在区域内
    """
    # 将区域转换为像素坐标
    rx1 = region[0] * img_width
    ry1 = region[1] * img_height
    rx2 = region[2] * img_width
    ry2 = region[3] * img_height

    # 计算交集
    ix1 = max(bbox[0], rx1)
    iy1 = max(bbox[1], ry1)
    ix2 = min(bbox[2], rx2)
    iy2 = min(bbox[3], ry2)

    if ix1 >= ix2 or iy1 >= iy2:
        return False

    intersection = (ix2 - ix1) * (iy2 - iy1)
    box_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

    overlap_ratio = intersection / box_area if box_area > 0 else 0
    return overlap_ratio >= overlap_threshold


def infer_region_from_net(detections, img_width, img_height, expand_ratio=0.3):
    """
    根据安全网位置推断护轮坎区域

    安全网通常设置在护轮坎边缘，因此可以用net的位置来定义危险区域

    Args:
        detections: 检测结果列表
        img_width, img_height: 图像尺寸
        expand_ratio: 区域扩展比例

    Returns:
        list or None: 推断的区域坐标 [x1, y1, x2, y2] 归一化坐标，或None
    """
    net_boxes = []
    for class_id, conf, bbox in detections:
        if class_id == CLASS_NET:
            net_boxes.append(bbox)

    if not net_boxes:
        return None

    # 合并所有net的边界框
    all_x1 = min(b[0] for b in net_boxes)
    all_y1 = min(b[1] for b in net_boxes)
    all_x2 = max(b[2] for b in net_boxes)
    all_y2 = max(b[3] for b in net_boxes)

    # 扩展区域（护轮坎区域比net稍大）
    width = all_x2 - all_x1
    height = all_y2 - all_y1

    expanded_x1 = max(0, all_x1 - width * expand_ratio)
    expanded_y1 = max(0, all_y1 - height * expand_ratio)
    expanded_x2 = min(img_width, all_x2 + width * expand_ratio)
    expanded_y2 = min(img_height, all_y2 + height * expand_ratio)

    # 返回归一化坐标
    return [
        expanded_x1 / img_width,
        expanded_y1 / img_height,
        expanded_x2 / img_width,
        expanded_y2 / img_height
    ]


def check_alarm(detections, region, img_width, img_height, auto_region=True):
    """
    根据检测结果判断是否报警

    Args:
        detections: 检测结果列表，每个元素为 (class_id, confidence, bbox)
        region: 护轮坎区域 [x1, y1, x2, y2] 归一化坐标 (默认/备用)
        img_width, img_height: 图像尺寸
        auto_region: 是否自动根据net位置推断区域

    Returns:
        tuple: (is_alarm, reason, intruder_boxes, actual_region)
            - is_alarm: int, 1=报警, 0=不报警
            - reason: str, 报警/不报警原因
            - intruder_boxes: list, 闯入人员的检测框坐标列表
            - actual_region: list, 实际使用的区域坐标
    """
    # 尝试从net位置推断区域
    actual_region = region
    region_source = "default"

    if auto_region:
        inferred = infer_region_from_net(detections, img_width, img_height)
        if inferred:
            actual_region = inferred
            region_source = "inferred_from_net"

    has_net = False
    has_equipment = False
    persons_in_region = []  # 在护轮坎区域内的人员

    for class_id, conf, bbox in detections:
        if class_id == CLASS_NET:
            has_net = True
        elif class_id == CLASS_EQUIPMENT:
            has_equipment = True
        elif class_id == CLASS_PERSON:
            # 检查人员是否在护轮坎区域内
            if box_in_region(bbox, actual_region, img_width, img_height):
                persons_in_region.append({
                    "bbox": [round(x, 2) for x in bbox],
                    "confidence": round(conf, 4)
                })

    # 报警逻辑
    if len(persons_in_region) == 0:
        return 0, "护轮坎区域内未检测到人员，无需报警", [], actual_region, region_source

    if has_net and has_equipment:
        return 0, "检测到人员，但安全网和救生衣均已设置，无需报警", [], actual_region, region_source

    # 需要报警
    missing = []
    if not has_net:
        missing.append("安全网")
    if not has_equipment:
        missing.append("救生衣")

    reason = f"警报！护轮坎区域检测到{len(persons_in_region)}人闯入，缺少: {', '.join(missing)}"
    return 1, reason, persons_in_region, actual_region, region_source


def process_image(model, image_path, region, conf_threshold=0.5, save_result=True, output_dir=None, auto_region=True):
    """
    处理单张图片

    Args:
        model: YOLO模型
        image_path: 图片路径
        region: 护轮坎区域坐标 (默认/备用)
        conf_threshold: 置信度阈值
        save_result: 是否保存检测结果图片
        output_dir: 结果保存目录
        auto_region: 是否自动根据net位置推断区域

    Returns:
        dict: 检测结果
    """
    image_path = Path(image_path)

    # 读取图片获取尺寸
    img = cv2.imread(str(image_path))
    if img is None:
        return {"error": f"无法读取图片: {image_path}"}
    img_height, img_width = img.shape[:2]

    # 推理
    results = model(str(image_path), conf=conf_threshold, verbose=False)[0]

    # 解析检测结果
    detections = []
    for box in results.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        bbox = box.xyxy[0].tolist()
        detections.append((class_id, confidence, bbox))

    # 判断是否报警
    is_alarm, reason, intruder_boxes, actual_region, region_source = check_alarm(
        detections, region, img_width, img_height, auto_region
    )

    # 构建结果
    result = {
        "image": str(image_path),
        "timestamp": datetime.now().isoformat(),
        "alarm": is_alarm,  # 二分类: 1=报警, 0=不报警
        "reason": reason,
        "intruders": intruder_boxes,  # 闯入人员的检测框
        "region": {
            "normalized": actual_region,
            "pixel": [
                round(actual_region[0] * img_width, 2),
                round(actual_region[1] * img_height, 2),
                round(actual_region[2] * img_width, 2),
                round(actual_region[3] * img_height, 2)
            ],
            "source": region_source  # "default" 或 "inferred_from_net"
        },
        "all_detections": [
            {
                "class": CLASS_NAMES[d[0]],
                "confidence": round(d[1], 4),
                "bbox": [round(x, 2) for x in d[2]]
            }
            for d in detections
        ],
        "statistics": {
            "person_count": sum(1 for d in detections if d[0] == CLASS_PERSON),
            "person_in_region": len(intruder_boxes),
            "net_detected": any(d[0] == CLASS_NET for d in detections),
            "equipment_detected": any(d[0] == CLASS_EQUIPMENT for d in detections)
        }
    }

    # 保存结果图片
    if save_result:
        if output_dir is None:
            output_dir = Path(__file__).parent / "output"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 绘制护轮坎区域
        rx1 = int(actual_region[0] * img_width)
        ry1 = int(actual_region[1] * img_height)
        rx2 = int(actual_region[2] * img_width)
        ry2 = int(actual_region[3] * img_height)
        region_color = (0, 255, 255) if region_source == "default" else (255, 255, 0)  # 黄色/青色
        cv2.rectangle(img, (rx1, ry1), (rx2, ry2), region_color, 2)
        region_label = "Region" if region_source == "default" else "Region(auto)"
        cv2.putText(img, region_label, (rx1, ry1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, region_color, 2)

        # 绘制检测框
        colors = {
            CLASS_PERSON: (0, 0, 255),     # 红色 - 人
            CLASS_NET: (0, 255, 0),        # 绿色 - 安全网
            CLASS_EQUIPMENT: (255, 0, 0)   # 蓝色 - 救生衣
        }

        for class_id, conf, bbox in detections:
            x1, y1, x2, y2 = map(int, bbox)
            color = colors.get(class_id, (255, 255, 255))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"{CLASS_NAMES[class_id]}: {conf:.2f}"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 添加报警状态
        status_color = (0, 0, 255) if is_alarm else (0, 255, 0)
        status_text = "ALARM (1)" if is_alarm else "SAFE (0)"
        cv2.putText(img, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, status_color, 3)

        # 保存
        output_path = output_dir / f"result_{image_path.name}"
        cv2.imwrite(str(output_path), img)
        result["output_image"] = str(output_path)

    return result


def process_directory(model, input_dir, region, conf_threshold=0.5, save_result=True, output_dir=None, auto_region=True):
    """处理目录下的所有图片"""
    input_dir = Path(input_dir)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

    results = []
    image_files = sorted([f for f in input_dir.iterdir() if f.suffix.lower() in image_extensions])

    print(f"找到 {len(image_files)} 张图片")

    alarm_count = 0
    for i, image_path in enumerate(image_files):
        result = process_image(model, image_path, region, conf_threshold, save_result, output_dir, auto_region)
        results.append(result)

        if result.get("alarm", 0) == 1:
            alarm_count += 1
            intruder_count = len(result.get("intruders", []))
            print(f"[{i+1}/{len(image_files)}] {image_path.name}: ALARM(1) - {intruder_count}人闯入")
        else:
            print(f"[{i+1}/{len(image_files)}] {image_path.name}: SAFE(0)")

    print(f"\n处理完成: {len(image_files)} 张图片, {alarm_count} 次报警")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="护轮坎人员闯入检测 - 二分类输出",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 检测单张图片
  python process.py --image test.jpg

  # 指定护轮坎区域 (归一化坐标: x1,y1,x2,y2)
  python process.py --image test.jpg --region "0.1,0.2,0.5,0.8"

  # 检测目录下所有图片
  python process.py --dir ./images

  # 输出JSON结果
  python process.py --dir ./images --json results.json

输出格式:
  alarm: 1=报警, 0=不报警
  intruders: 闯入人员的检测框坐标列表 [{"bbox": [x1,y1,x2,y2], "confidence": 0.95}, ...]
        """
    )

    # 输入选项
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", type=str, help="输入图片路径")
    input_group.add_argument("--dir", type=str, help="输入图片目录")

    # 区域选项
    parser.add_argument("--region", type=str, default=None,
                       help="护轮坎区域，格式: x1,y1,x2,y2 (归一化坐标0-1)")
    parser.add_argument("--no-auto-region", action="store_true",
                       help="禁用自动区域推断（默认根据net位置自动推断区域）")

    # 模型选项
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL),
                       help=f"模型路径 (默认: {DEFAULT_MODEL})")
    parser.add_argument("--conf", type=float, default=0.5,
                       help="置信度阈值 (默认: 0.5)")

    # 输出选项
    parser.add_argument("--output", type=str, help="结果图片保存目录")
    parser.add_argument("--no-save", action="store_true", help="不保存结果图片")
    parser.add_argument("--json", type=str, help="保存JSON格式结果到文件")

    args = parser.parse_args()

    # 解析区域
    region = parse_region(args.region)
    auto_region = not args.no_auto_region
    print(f"护轮坎区域(默认): {region}")
    print(f"自动区域推断: {'启用' if auto_region else '禁用'}")

    # 加载模型
    model = load_model(args.model)

    # 处理
    if args.image:
        result = process_image(
            model, args.image, region,
            conf_threshold=args.conf,
            save_result=not args.no_save,
            output_dir=args.output,
            auto_region=auto_region
        )
        results = [result]

        # 打印结果
        print("\n" + "=" * 60)
        print(f"图片: {result['image']}")
        print(f"报警 (二分类): {result['alarm']}")
        print(f"原因: {result['reason']}")
        print(f"护轮坎区域: {result['region']['pixel']} ({result['region']['source']})")
        if result['intruders']:
            print(f"闯入人员检测框:")
            for i, intruder in enumerate(result['intruders']):
                print(f"  [{i+1}] bbox={intruder['bbox']}, conf={intruder['confidence']}")
        print(f"检测统计: 总人数={result['statistics']['person_count']}, "
              f"区域内={result['statistics']['person_in_region']}, "
              f"安全网={'有' if result['statistics']['net_detected'] else '无'}, "
              f"救生衣={'有' if result['statistics']['equipment_detected'] else '无'}")
        if 'output_image' in result:
            print(f"结果图片: {result['output_image']}")
        print("=" * 60)

    else:
        results = process_directory(
            model, args.dir, region,
            conf_threshold=args.conf,
            save_result=not args.no_save,
            output_dir=args.output,
            auto_region=auto_region
        )

    # 保存JSON结果
    if args.json:
        with open(args.json, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nJSON结果已保存: {args.json}")

    # 返回报警状态
    has_alarm = any(r.get("alarm", 0) == 1 for r in results)
    return 1 if has_alarm else 0


if __name__ == "__main__":
    exit(main())
