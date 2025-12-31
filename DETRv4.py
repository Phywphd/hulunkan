import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import sys
from pathlib import Path

# ================= 1. RT-DETRv4 模拟类 (兼容 Ultralytics) =================
class RTv4SingleBox:
    def __init__(self, cls, conf, xyxy):
        self.cls = [cls]
        self.conf = [conf]
        self.xyxy = [xyxy] # Tensor [x1, y1, x2, y2]

class RTv4Boxes:
    def __init__(self, cls, conf, xyxy):
        self.cls_tensor = cls
        self.conf_tensor = conf
        self.xyxy_tensor = xyxy

    def __len__(self):
        return len(self.cls_tensor)

    def __iter__(self):
        for i in range(len(self.cls_tensor)):
            yield RTv4SingleBox(self.cls_tensor[i], self.conf_tensor[i], self.xyxy_tensor[i])

class RTv4Results:
    def __init__(self, boxes):
        self.boxes = boxes

# ================= 2. RT-DETRv4 包装器 =================
class RTDETRv4Wrapper:
    def __init__(self, repo_path, config_path, pth_path, device='cuda'):
        # 动态添加仓库路径以确保能导入对应的 engine 或 src
        repo_path = str(Path(repo_path).resolve())
        if repo_path not in sys.path:
            sys.path.append(repo_path)
        
        # 导入仓库内的配置加载工具
        try:
            from rtdetrv4_repo.engine.core import YAMLConfig 
        except ImportError:
            from src.core import YAMLConfig # 部分版本可能在 src 下

        self.device = torch.device(device)
        
        # 加载配置与模型逻辑 (参考 DEIM 加载逻辑)
        cfg = YAMLConfig(config_path, resume=pth_path)
        if 'HGNetv2' in cfg.yaml_cfg: 
            cfg.yaml_cfg['HGNetv2']['pretrained'] = False
        
        checkpoint = torch.load(pth_path, map_location='cpu')
        state = checkpoint['ema']['module'] if 'ema' in checkpoint else checkpoint['model']
        cfg.model.load_state_dict(state)
        
        # 封装推理引擎
        class InferenceEngine(nn.Module):
            def __init__(self, base_model, postprocessor):
                super().__init__()
                self.model = base_model.deploy()
                self.postprocessor = postprocessor.deploy()
            def forward(self, images, orig_target_sizes):
                return self.postprocessor(self.model(images), orig_target_sizes)

        self.engine = InferenceEngine(cfg.model, cfg.postprocessor).to(self.device).eval()
        
        # 预设参数
        self.names = {0: 'person', 1: 'net', 2: 'equipment', 3: 'wheel_guard'}
        self.size = tuple(cfg.yaml_cfg.get('eval_spatial_size', [640, 640]))
        self.vit_backbone = 'DINO' in str(cfg.yaml_cfg)
        
        self.transforms = T.Compose([
            T.Resize(self.size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
            if self.vit_backbone else T.Lambda(lambda x: x)
        ])

    def predict(self, source, conf=0.25, **kwargs):
        """统一推理接口"""
        im_pil = Image.open(str(source)).convert('RGB')
        w, h = im_pil.size
        orig_size = torch.tensor([[w, h]]).to(self.device)
        im_data = self.transforms(im_pil).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # 获取 Batch 中的第一张图结果
            labels, boxes, scores = self.engine(im_data, orig_size)
            
        mask = scores[0] > conf
        # 模拟 Ultralytics 返回 Results 列表格式
        return [RTv4Results(RTv4Boxes(labels[0][mask], scores[0][mask], boxes[0][mask]))]