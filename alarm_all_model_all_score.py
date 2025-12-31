#!/usr/bin/env python3
"""
码头护轮坎闯入检测 - 多模型对比测试脚本
功能：调用 Logic A/B/C 进行图像分类。
"""
from ultralytics import RTDETR, YOLO
from pathlib import Path
import cv2
import sys
from DEIM import DEIMWrapper
from DETRv4 import RTDETRv4Wrapper
import time


# ================= 配置区 =================
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
DRAW_MARKERS = False    #是否在图像中标注
LOGIC = 'C'   #A, B, C
DATA_YAML = "data.yaml"
# 测试图像目录
ALARM_IMG_DIR = Path("alarm_image")    # 正样本：实际应该报警的图
NORMAL_IMG_DIR = Path("normal_image")  # 负样本：实际不该报警的图


# 所有模型路径
MODELS = {
    "rtdetr_v2": {
        "path": "weights/rtdetr_v2_best.pt",
        "type": "rtdetr"
    },
    "rtdetr_v4": {
        "path": "weights/best_stg2.pth",
        "type": "rtdetrv4",
        "config": "configs/rtv4_hgnetv2_l_hulunkan.yml",
        "repo_path": "rtdetrv4_repo",
        "output_dir": "rtdetrv4_hulunkan_test"
    },
    "deim": {
        "config": "configs/deimv2_dinov3_x_custom.yml",
        "path": "weights/best_stg2_slim.pth", 
        "type": "deim"
    },
    "yolo11m": {
        "path": "weights/yolo11m_best.pt",
        "type": "yolo"
    },
    "yolo13l": {
        "path": "weights/yolo13l_best.pt",
        "type": "yolo"
    },
}

def logic_A(img_path, model):

    # 1. 模型推理
    results = model.predict(source=str(img_path), save=False, conf=min(CLASS_THRESHOLDS.values()), verbose=False)
    r = results[0]
    img = cv2.imread(str(img_path))
    
    # 数据分类存储
    objs = {name: [] for name in CLASS_NAMES}
    
    # 2. 解析检测框
    for box in r.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if conf < CLASS_THRESHOLDS.get(cls_id, 0.5):
            continue
            
        name = CLASS_NAMES[cls_id]
        coords = box.xyxy[0].cpu().numpy().astype(int) # [x1, y1, x2, y2]
        objs[name].append({'coords': coords, 'conf': conf, 'cls_id': cls_id})

        # 绘制基础框
        color = CLASS_COLORS.get(cls_id, (255, 255, 255))
        cv2.rectangle(img, (coords[0], coords[1]), (coords[2], coords[3]), color, 2)
        cv2.putText(img, f"{name} {conf:.2f}", (coords[0], coords[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # 3. 执行报警判定逻辑
    is_alarm = False
    has_net = len(objs['net']) > 0
    
    for p in objs['person']:
        px1, py1, px2, py2 = p['coords']
        foot_point = ((px1 + px2) // 2, py2) # 脚部中点
        
        # 判定 A: 是否踩在任何一个护轮坎上
        on_guard = False
        for g in objs['wheel_guard']:
            gx1, gy1, gx2, gy2 = g['coords']
            if gx1 <= foot_point[0] <= gx2 and gy1 <= foot_point[1] <= gy2:
                on_guard = True
                break
        
        if on_guard:
            # 判定 B: 此人是否穿了救生衣 (equipment 中心点在人框内)
            has_jacket = False
            for e in objs['equipment']:
                ex1, ey1, ex2, ey2 = e['coords']
                ecx, ecy = (ex1 + ex2) // 2, (ey1 + ey2) // 2
                if px1 <= ecx <= px2 and py1 <= ecy <= py2:
                    has_jacket = True
                    break
            
            # 触发规则：在护轮坎上 且 (没网 或 没穿衣)
            if not (has_net and has_jacket):
                is_alarm = True
                # 在人头上标记红色警示
                if DRAW_MARKERS:
                    cv2.circle(img, foot_point, 5, (0, 0, 255), -1)
                    cv2.putText(img, "DANGER", (px1, py1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    if is_alarm and DRAW_MARKERS:
        cv2.putText(img, "ALARM STATUS: TRIGGERED", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
    
    summary = f"P:{len(objs['person'])} G:{len(objs['wheel_guard'])} N:{len(objs['net'])} E:{len(objs['equipment'])}"
    return is_alarm, img, summary
    
def logic_B(img_path, model):

    # 1. 模型推理
    results = model.predict(
        source=str(img_path), 
        save=False, 
        conf=min(CLASS_THRESHOLDS.values()), 
        verbose=False
    )
    r = results[0]
    img = cv2.imread(str(img_path))
    
    # 按类别对检测结果进行分组
    objs = {name: [] for name in CLASS_NAMES}
    for box in r.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if conf < CLASS_THRESHOLDS.get(cls_id, 0.5):
            continue
        
        name = CLASS_NAMES[cls_id]
        coords = box.xyxy[0].cpu().numpy().astype(int)
        objs[name].append({'coords': coords, 'conf': conf})

    # 2. 闯入判定逻辑 (Logic B)
    is_alarm = False
    alarm_count = 0
    
    for p in objs['person']:
        px1, py1, px2, py2 = p['coords']
        # 【关键：定义人脚部足迹区域】取人框底部的 20% 区域，增加判定鲁棒性
        foot_region = [px1, int(py2 - (py2-py1) * 0.2), px2, py2]
        
        on_guard = False
        target_guard_coords = None
        
        # A. 判定足迹区域与护轮坎的重叠度 (IoA)
        for g in objs['wheel_guard']:
            gx1, gy1, gx2, gy2 = g['coords']
            # 计算交集矩形
            ix1, iy1 = max(foot_region[0], gx1), max(foot_region[1], gy1)
            ix2, iy2 = min(foot_region[2], gx2), min(foot_region[3], gy2)
            
            if ix2 > ix1 and iy2 > iy1:
                inter_area = (ix2 - ix1) * (iy2 - iy1)
                foot_area = (foot_region[2] - foot_region[0]) * (foot_region[3] - foot_region[1])
                # 如果人脚部面积有 30% 以上落在护轮坎内，判定为踏入
                if inter_area / foot_area > 0.3:
                    on_guard = True
                    target_guard_coords = g['coords']
                    break
        
        if on_guard:
            # B. 局部安全网判定：网必须在当前护轮坎附近
            has_local_net = False
            gcx, gcy = (target_guard_coords[0] + target_guard_coords[2]) // 2, (target_guard_coords[1] + target_guard_coords[3]) // 2
            for n in objs['net']:
                nx1, ny1, nx2, ny2 = n['coords']
                ncx, ncy = (nx1 + nx2) // 2, (ny1 + ny2) // 2
                distance = ((gcx - ncx)**2 + (gcy - ncy)**2)**0.5
                if distance < 300: 
                    has_local_net = True
                    break
            
            # C. 救生衣判定：equipment 框中心在人框内
            has_jacket = False
            for e in objs['equipment']:
                ex1, ey1, ex2, ey2 = e['coords']
                ecx, ecy = (ex1 + ex2) // 2, (ey1 + ey2) // 2
                if px1 <= ecx <= px2 and py1 <= ecy <= py2:
                    has_jacket = True
                    break
            
            # D. 综合报警规则：踏入护轮坎 且 (没局部网 或 没穿救生衣)
            if not (has_local_net and has_jacket):
                is_alarm = True
                alarm_count += 1
                if DRAW_MARKERS:
                    # 绘制橙色高亮表示“风险脚部区域”
                    cv2.rectangle(img, (foot_region[0], foot_region[1]), (foot_region[2], foot_region[3]), (0, 165, 255), 2)
                    cv2.putText(img, "RISK: NO SAFETY", (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # 3. 绘制通用检测结果
    for name, instances in objs.items():
        for inst in instances:
            x1, y1, x2, y2 = inst['coords']
            color = (0, 255, 0) if name == 'person' else (255, 0, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
    if is_alarm and DRAW_MARKERS:
        cv2.putText(img, f"ALARM: {alarm_count} PERSON(S)", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    summary = f"P:{len(objs['person'])} G:{len(objs['wheel_guard'])} N:{len(objs['net'])} E:{len(objs['equipment'])}"
    return is_alarm, img, summary
    
def logic_C(img_path, model):
    """
    Logic C: 概率风险融合逻辑 (Probabilistic Risk Fusion)
    核心改进：引入空间缓冲区与加权风险评分，平衡漏报与误报。
    """
    # 1. 模型推理
    results = model.predict(source=str(img_path), save=False, conf=0.25, verbose=False)
    r = results[0]
    img = cv2.imread(str(img_path))
    
    # 类别映射
    names = model.names
    objs = {'person': [], 'net': [], 'equipment': [], 'wheel_guard': []}
    
    for box in r.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        name = names[cls_id]
        if name in objs:
            objs[name].append({
                'coords': box.xyxy[0].cpu().numpy().astype(int),
                'conf': conf
            })

    is_alarm = False
    alarm_reasons = []

    # 2. 全局/局部安全网强度
    net_strength = 0
    for n in objs['net']:
        net_strength = max(net_strength, n['conf'])

    # 3. 对每个人进行风险评分
    for p in objs['person']:
        px1, py1, px2, py2 = p['coords']
        risk_score = 0
        
        # A. 空间风险：人是否靠近/踏上护轮坎 (增加 10 像素缓冲区)
        on_guard = False
        foot_region = [px1, int(py2 - (py2-py1)*0.25), px2, py2 + 10] # 纵向向下延伸缓冲区
        
        for g in objs['wheel_guard']:
            gx1, gy1, gx2, gy2 = g['coords']
            # 扩大护轮坎感应区
            gx1, gy1, gx2, gy2 = gx1-10, gy1-10, gx2+10, gy2+10
            
            ix1, iy1 = max(foot_region[0], gx1), max(foot_region[1], gy1)
            ix2, iy2 = min(foot_region[2], gx2), min(foot_region[3], gy2)
            
            if ix2 > ix1 and iy2 > iy1:
                inter_area = (ix2 - ix1) * (iy2 - iy1)
                foot_area = (foot_region[2] - foot_region[0]) * (foot_region[3] - foot_region[1])
                if inter_area / foot_area > 0.15: # 降低重叠门槛至 15%
                    on_guard = True
                    break
        
        if on_guard:
            # B. 装备风险评分 (Equipment Score)
            # 检查人身上是否有救生衣，若无，风险值增加
            has_jacket = False
            for e in objs['equipment']:
                ex1, ey1, ex2, ey2 = e['coords']
                # 判断救生衣是否在人框的上半部
                if px1 <= (ex1+ex2)/2 <= px2 and py1 <= (ey1+ey2)/2 <= py1 + (py2-py1)*0.6:
                    has_jacket = True
                    break
            
            # C. 决策融合：保守判定策略
            # 如果在护轮坎上且 (没有救生衣 OR 没有网)，风险得分跨越阈值
            if not has_jacket: risk_score += 0.6
            if net_strength < 0.4: risk_score += 0.5 # 网的置信度太低也视为风险
            
            if risk_score >= 0.5:
                is_alarm = True
                if True: # DRAW_MARKERS
                    color = (0, 0, 255) # 红色
                    cv2.rectangle(img, (px1, py1), (px2, py2), color, 2)
                    cv2.putText(img, f"RISK:{risk_score:.1f}", (px1, py1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    summary = f"LogicC -> P:{len(objs['person'])} G:{len(objs['wheel_guard'])} Alarm:{is_alarm}"
    return is_alarm, img, summary


# ================= 工具函数 =================

def fix_model_attn(model):
    for m in model.modules():
        if m.__class__.__name__ == 'AAttn':
            if hasattr(m, 'qk') and not hasattr(m, 'qkv'):
                m.qkv = m.qk
            elif hasattr(m, 'qkv') and not hasattr(m, 'qk'):
                m.qk = m.qkv
    return model

def run_logic_test(model_name, config, logic_func):
    """运行特定模型的闯入检测并计算混淆矩阵指标"""
    if config["type"] == "rtdetr":
        model = RTDETR(config["path"])
        model = fix_model_attn(model)
    elif config["type"] == "yolo":
        model = YOLO(config["path"])
        model = fix_model_attn(model)
    elif config["type"] == "deim":
        model = DEIMWrapper(config["config"], config["path"])
    elif config["type"] == "rtdetrv4":
        model = RTDETRv4Wrapper(
            repo_path=config["repo_path"],
            config_path=config["config"],
            pth_path=config["path"]
        )
    
    # 混淆矩阵计数
    tp, fn, fp, tn = 0, 0, 0, 0
    
    # 1. 处理报警图像 (Positive Samples)
    alarm_imgs = list(ALARM_IMG_DIR.glob('*.[jp][pn][g]')) + list(ALARM_IMG_DIR.glob('*.jpeg'))
    normal_imgs = list(NORMAL_IMG_DIR.glob('*.[jp][pn][g]')) + list(NORMAL_IMG_DIR.glob('*.jpeg'))
    
    start_time = time.perf_counter()
    
    print(f"正在处理模型 {model_name} 的报警图像 ({len(alarm_imgs)}张)...")
    for img_path in alarm_imgs:
        is_alarm, _, _ = logic_func(img_path, model)
        if is_alarm:
            tp += 1  # 预测报警，实际报警 -> True Positive
        else:
            fn += 1  # 预测正常，实际报警 -> False Negative (漏报)

    # 2. 处理正常图像 (Negative Samples)
    print(f"正在处理模型 {model_name} 的正常图像 ({len(normal_imgs)}张)...")
    for img_path in normal_imgs:
        is_alarm, _, _ = logic_func(img_path, model)
        if is_alarm:
            fp += 1  # 预测报警，实际正常 -> False Positive (误报)
        else:
            tn += 1  # 预测正常，实际正常 -> True Negative
            
    end_time = time.perf_counter()
    # 计算平均每张图的延迟 (ms)
    avg_latency = ((end_time - start_time) / (len(alarm_imgs) + len(normal_imgs))) * 1000

    # 3. 计算指标
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "name": model_name,
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "acc": accuracy, "pre": precision, "rec": recall, "f1": f1, "speed": avg_latency
    }

# ================= 主程序 =================

def main():
    if not ALARM_IMG_DIR.exists() or not NORMAL_IMG_DIR.exists():
        print("错误: 找不到 alarm_image 或 normal_image 目录")
        return

    all_stats = []
    
    # 根据全局变量 LOGIC 选择逻辑函数
    if LOGIC == 'A':
        current_logic = logic_A
    elif LOGIC == 'B':
        current_logic = logic_B
    else:
        current_logic = logic_C

    for name, config in MODELS.items():
       try:
           stat = run_logic_test(name, config, current_logic)
           all_stats.append(stat)
       except Exception as e:
           print(f"模型 {name} 测试失败: {e}")

    # --- 终端输出汇总表格 ---
    print("\n" + "=" * 130)
    header = f"{'模型(Model)':<15} | {'TP(召回)':<8} | {'FN(漏报)':<8} | {'FP(误报)':<8} | {'TN(正确)':<8} | {'准确率(Acc)':<10} | {'精确率(Pre)':<10} | {'召回率(Rec)':<10} | {'F1-Score':<10} | {'Speed(ms)':<10}"
    print(header)
    print("-" * 130)
    
    for s in all_stats:
        print(f"{s['name']:<15} | {s['tp']:<8} | {s['fn']:<8} | {s['fp']:<8} | {s['tn']:<8} | "
              f"{s['acc']:<10.4f} | {s['pre']:<10.4f} | {s['rec']:<10.4f} | {s['f1']:<10.4f} | {s['speed']:<10.2f}")
    
    print("=" * 130)
    print(f"当前使用的测试逻辑: Logic {LOGIC}")
    print("测试完成。")

if __name__ == "__main__":
    main()