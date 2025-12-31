#!/usr/bin/env python3
"""
在 test 集上测试目标检测效果
"""
from ultralytics import RTDETR
from pathlib import Path
import random
import cv2

# 配置 - 使用相对路径
MODEL_PATH = "weights/rtdetr_v2_best.pt"
DATA_YAML = "data.yaml"
TEST_DIR = Path("public/home/sjtu_wumei/ai4sci_work/test/images")  # 需要自行准备测试图片
OUTPUT_DIR = Path("test_results")

# 不同类别的置信度阈值
CLASS_NAMES = ['person', 'net', 'equipment', 'wheel_guard']
CLASS_THRESHOLDS = {
    0: 0.5,    # person
    1: 0.5,    # net
    2: 0.3,    # equipment - 降低阈值，提高召回率
    3: 0.5,    # wheel_guard
}

# 可视化颜色 (BGR)
CLASS_COLORS = {
    0: (0, 255, 0),    # person - 绿色
    1: (255, 0, 0),    # net - 蓝色
    2: (0, 255, 255),  # equipment - 黄色
    3: (0, 0, 255),    # wheel_guard - 红色
}

def evaluate_on_test():
    """在 test 集上评估模型"""
    print("=" * 50)
    print("在 test 集上评估模型")
    print("=" * 50)

    model = RTDETR(MODEL_PATH)
    results = model.val(data=DATA_YAML, split='test')

    print(f"\n测试集结果:")
    print(f"  mAP50: {results.box.map50:.4f}")
    print(f"  mAP50-95: {results.box.map:.4f}")

    return results
    
def check_intrusion(r, CLASS_NAMES):
    """
    闯入逻辑判定函数
    返回: (is_alarm, alarm_message, person_status_list)
    """
    # 1. 类别分组
    persons = []
    guards = []
    nets = []
    equipments = []

    for box in r.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        # 使用组员定义的阈值过滤逻辑（这里简化为0.5，实际运行时会继承你代码里的过滤）
        if conf < 0.4: continue 
        
        name = CLASS_NAMES[cls]
        coords = box.xyxy[0].cpu().numpy().astype(int) # [x1, y1, x2, y2]
        
        if name == 'person': persons.append(coords)
        elif name == 'wheel_guard': guards.append(coords)
        elif name == 'net': nets.append(coords)
        elif name == 'equipment': equipments.append(coords)

    # 2. 全局状态：是否有安全网 (Net)
    # 逻辑：如果图片中检测到net，视为安全设施已就位
    has_net = len(nets) > 0
    
    is_alarm = False
    alarm_reasons = []
    
    # 3. 针对每个人进行判定
    for p_box in persons:
        px1, py1, px2, py2 = p_box
        # 计算脚部参考点：底边中点
        foot_x = (px1 + px2) // 2
        foot_y = py2
        
        # A. 判定是否踩在护轮坎(wheel_guard)上
        is_on_guard = False
        for g_box in guards:
            gx1, gy1, gx2, gy2 = g_box
            # 允许 5 像素的容错偏差
            if gx1 - 5 <= foot_x <= gx2 + 5 and gy1 - 5 <= foot_y <= gy2 + 5:
                is_on_guard = True
                break
        
        if is_on_guard:
            # B. 判定此人是否穿了救生衣 (equipment)
            # 逻辑：判断是否有 equipment 框落在该 person 框的范围内 (IoA判定)
            has_jacket = False
            for e_box in equipments:
                ex1, ey1, ex2, ey2 = e_box
                # 计算救生衣中心点
                ecx, ecy = (ex1 + ex2) // 2, (ey1 + ey2) // 2
                # 如果救生衣中心在人框内，视为该人穿戴
                if px1 <= ecx <= px2 and py1 <= ecy <= py2:
                    has_jacket = True
                    break
            
            # C. 最终报警逻辑判定
            # 规则：在护轮坎上时，(未穿救生衣) OR (未设置安全网) -> 报警
            if not (has_jacket and has_net):
                is_alarm = True
                reason = f"人员在护轮坎内且安全措施不足(救生衣:{has_jacket}, 安全网:{has_net})"
                alarm_reasons.append(reason)

    return is_alarm, alarm_reasons


def visualize_predictions(num_samples=10):
    """可视化预测结果 - 支持不同类别不同阈值，逐张处理避免OOM"""
    print("\n" + "=" * 50)
    if num_samples == -1:
        print("可视化全部测试图片的预测结果")
    else:
        print(f"可视化 {num_samples} 张测试图片的预测结果")
    print("=" * 50)
    print(f"类别阈值: {CLASS_THRESHOLDS}")

    output_dir = OUTPUT_DIR / "predictions_v2"
    output_dir_alarm = output_dir / "alarm"
    output_dir_regular = output_dir / "regular"
    
    output_dir_alarm.mkdir(parents=True, exist_ok=True)
    output_dir_regular.mkdir(parents=True, exist_ok=True)

    model = RTDETR(MODEL_PATH)

    # 获取图片列表
    images = list(TEST_DIR.glob('*.jpeg')) + list(TEST_DIR.glob('*.jpg')) + list(TEST_DIR.glob('*.png'))

    # num_samples=-1 表示全部
    if num_samples == -1:
        samples = sorted(images)
    else:
        random.seed(42)
        samples = random.sample(images, min(num_samples, len(images)))

    print(f"共 {len(samples)} 张图片待处理\n")

    # 用最低阈值推理
    min_conf = min(CLASS_THRESHOLDS.values())

    # 逐张处理，避免显存溢出
    for i, img_path in enumerate(samples):
        # 单张推理
        results = model.predict(
            source=str(img_path),
            save=False,
            conf=min_conf,
            verbose=False,
        )
        r = results[0]

        img = cv2.imread(str(img_path))
        
        is_alarm, reasons = check_intrusion(r, CLASS_NAMES)

        detections = []
        # 按类别阈值过滤并绘制
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            # 检查是否超过该类别的阈值
            if conf < CLASS_THRESHOLDS.get(cls, 0.5):
                continue

            cls_name = CLASS_NAMES[cls]
            color = CLASS_COLORS.get(cls, (255, 255, 255))

            # 获取边界框坐标
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            # 绘制边界框
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # 绘制标签
            label = f"{cls_name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
            cv2.putText(img, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

            detections.append(f"{cls_name}:{conf:.2f}")
            
        if is_alarm:
            cv2.putText(img, "!!! ALARM: INTRUSION DETECTED !!!", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            print(f"图片 {img_path.name} 触发报警: {reasons}")

            # 保存结果
            output_path = output_dir_alarm / f"{img_path.stem}_detected.jpg"
            cv2.imwrite(str(output_path), img)
        else:
            # 保存结果
            output_path = output_dir_regular / f"{img_path.stem}_detected.jpg"
            cv2.imwrite(str(output_path), img)

        # 简洁输出进度
        det_str = ", ".join(detections) if detections else "无检测"
        print(f"[{i+1}/{len(samples)}] {img_path.name}: {det_str}")

    print(f"\n可视化结果保存在: {output_dir}")
    return


if __name__ == "__main__":
    # 1. 评估指标
    # evaluate_on_test()

    # 2. 可视化预测 (-1 表示全部)
    visualize_predictions(num_samples=-1)
