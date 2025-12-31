#!/usr/bin/env python3
import torch
import cv2
import sys
import numpy as np
from pathlib import Path

# ================= 1. 环境准备：导入 RT-DETRv4 源码库 =================
# 基于你提供的路径
REPO_PATH = "/home/kemove/ai4s_tmp/rtdetrv4_repo"
sys.path.append(REPO_PATH)

try:
    # 常见的 RT-DETR 仓库结构导入方式
    from src.core import YAMLConfig
except ImportError:
    print(f"错误: 无法在 {REPO_PATH} 中找到 src.core。请检查 repo_path 是否正确。")
    sys.exit(1)

# ================= 2. 导入你本地的逻辑函数 =================
# 假设你的函数在同一目录下的 logic_script.py 中，请根据实际情况修改
# from logic_script import logic_A, logic_B
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

    # 3. 执行报警判定逻辑 (此处可随时替换)
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
            # B. 局部安全网判定：网必须在当前护轮坎附近（距离阈值可调，此处设为 300 像素）
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

# ================= 3. RT-DETRv4 兼容性包装类 =================

class RTDETRv4Wrapper:
    """将 RT-DETRv4 包装成类似 Ultralytics 的接口，以便适配 logic_A/B"""
    def __init__(self, cfg_path, weight_path, device='cuda'):
        self.cfg = YAMLConfig(cfg_path, resume=weight_path)
        self.model = self.cfg.model.deploy()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        # 加载权重
        checkpoint = torch.load(weight_path, map_location='cpu')
        if 'model' in checkpoint:
            self.model.load_state_dict(checkpoint['model'])
        else:
            self.model.load_state_dict(checkpoint)
        print(f"RT-DETRv4 模型加载成功: {weight_path}")

    def predict(self, source, conf=0.3, **kwargs):
        """模拟 Ultralytics 的 predict 接口"""
        img0 = cv2.imread(str(source))
        h, w = img0.shape[:2]
        
        # 基础预处理（假设模型输入为 640x640，根据实际 config 调整）
        img = cv2.resize(img0, (640, 640))
        img = img.transpose(2, 0, 1)  # HWC to CHW
        img = torch.from_numpy(img).to(self.device).float() / 255.0
        img = img.unsqueeze(0)
        
        # 推理
        with torch.no_grad():
            outputs = self.model(img) # RT-DETR 通常返回 scores, boxes
        
        # 结果解析（此处需要根据你的具体模型输出调整，通常 v4 返回 dict）
        # 这里模拟包装成类 Ultralytics Results 对象
        class Results:
            def __init__(self, boxes_data):
                self.boxes = boxes_data
        
        class Boxes:
            def __init__(self, xyxy, conf, cls):
                self.xyxy = [torch.tensor(xyxy)]
                self.conf = [torch.tensor(conf)]
                self.cls = [torch.tensor(cls)]

        # 过滤低置信度（示例逻辑，根据模型输出结构微调）
        # 假设 outputs 包含 'labels', 'boxes', 'scores'
        labels = outputs['labels'][0].cpu().numpy()
        scores = outputs['scores'][0].cpu().numpy()
        boxes = outputs['boxes'][0].cpu().numpy() # 归一化坐标 [cx, cy, w, h]
        
        keep = scores > conf
        final_boxes = []
        final_conf = []
        final_cls = []
        
        for i in range(len(keep)):
            if keep[i]:
                # 坐标转换: cxcywh -> xyxy 且反归一化到原图
                cx, cy, bw, bh = boxes[i]
                x1 = (cx - bw/2) * w
                y1 = (cy - bh/2) * h
                x2 = (cx + bw/2) * w
                y2 = (cy + bh/2) * h
                final_boxes.append([x1, y1, x2, y2])
                final_conf.append(scores[i])
                final_cls.append(labels[i])
        
        if not final_boxes: # 防止空检测报错
            return [Results(Boxes(np.zeros((0,4)), np.zeros(0), np.zeros(0)))]
            
        return [Results(Boxes(np.array(final_boxes), np.array(final_conf), np.array(final_cls)))]

# ================= 4. 批量运行与统计 =================

def run_v4_logic_test():
    # 配置路径
    cfg_path = "/home/kemove/ai4s_tmp/rtdetrv4_repo/configs/rtv4/rtv4_hgnetv2_l_hulunkan.yml"
    weight_path = "/home/kemove/ai4s_tmp/rtdetrv4_repo/outputs/rtv4_hgnetv2_l_hulunkan/best_stg2.pth"
    test_img_dir = Path("/home/kemove/ai4s_tmp/hulunkan_detection/coco_dataset/test")

    # 初始化包装好的模型
    v4_model = RTDETRv4Wrapper(cfg_path, weight_path)

    img_list = list(test_img_dir.glob('*.jpg')) + list(test_img_dir.glob('*.png'))
    stats = {"alarm": 0, "normal": 0}

    print(f"开始对 {len(img_list)} 张图片运行 RT-DETRv4 逻辑检测...")

    for img_p in img_list:
        # 调用你的 logic_B (它会内部调用 v4_model.predict)
        # 确保你本地的 logic_B 已经导入
        try:
            is_alarm, _, _ = logic_B(img_p, v4_model)
            if is_alarm:
                stats["alarm"] += 1
            else:
                stats["normal"] += 1
        except Exception as e:
            print(f"图片 {img_p.name} 处理出错: {e}")

    # --- 输出统计表格 ---
    print("\n" + "=" * 45)
    print(f"{'RT-DETRv4 闯入检测统计汇总':^45}")
    print("-" * 45)
    print(f"{'分类类型 (Category)':<25} | {'数量 (Count)':<15}")
    print("-" * 45)
    print(f"{'警报图像 (Alarm)':<25} | {stats['alarm']:<15}")
    print(f"{'正常图像 (Normal)':<25} | {stats['normal']:<15}")
    print("-" * 45)
    print(f"{'总计 (Total)':<25} | {len(img_list):<15}")
    print("=" * 45)

if __name__ == "__main__":
    # 注意：运行前请确保 logic_B 在作用域内可用
    # 如果 logic_B 定义在当前脚本，请确保它在 main 之前
    run_v4_logic_test()