import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import numpy as np

# ================= 修正后的 DEIM 模拟类 =================
class DEIMSingleBox:
    """
    模拟单条检测结果，确保 box.cls[0], box.conf[0], box.xyxy[0].cpu() 都能正常工作
    """
    def __init__(self, cls, conf, xyxy):
        # 包装成列表或 Tensor，以支持逻辑函数中的 [0] 访问
        self.cls = [cls]
        self.conf = [conf]
        self.xyxy = [xyxy] # 这里 xyxy 已经是 Tensor

class DEIMBoxes:
    """
    模拟 Ultralytics 的 r.boxes 容器
    """
    def __init__(self, cls, conf, xyxy):
        self.cls_tensor = cls
        self.conf_tensor = conf
        self.xyxy_tensor = xyxy

    def __len__(self):
        return len(self.cls_tensor)

    def __iter__(self):
        # 实现迭代器，让 for box in r.boxes 可以运行
        for i in range(len(self.cls_tensor)):
            yield DEIMSingleBox(
                self.cls_tensor[i], 
                self.conf_tensor[i], 
                self.xyxy_tensor[i]
            )

class DEIMResults:
    def __init__(self, boxes):
        self.boxes = boxes

# ================= DEIM 包装器 =================
class DEIMWrapper:
    def __init__(self, config_path, pth_path, device='cuda'):
        # ... (这里保持之前的 load_model 逻辑不变) ...
        # 确保加载了你同学的 engine.core 等
        from engine.core import YAMLConfig 
        cfg = YAMLConfig(config_path, resume=pth_path)
        
        # 简单的模型初始化逻辑封装
        if 'HGNetv2' in cfg.yaml_cfg: 
            cfg.yaml_cfg['HGNetv2']['pretrained'] = False
        checkpoint = torch.load(pth_path, map_location='cpu')
        state = checkpoint['ema']['module'] if 'ema' in checkpoint else checkpoint['model']
        cfg.model.load_state_dict(state)
        
        class InternalModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = cfg.model.deploy()
                self.postprocessor = cfg.postprocessor.deploy()
            def forward(self, images, orig_target_sizes):
                return self.postprocessor(self.model(images), orig_target_sizes)

        self.model = InternalModel().to(device).eval()
        self.device = device
        self.names = {0: 'person', 1: 'net', 2: 'equipment', 3: 'wheel_guard'}
        self.size = tuple(cfg.yaml_cfg.get('eval_spatial_size', [640, 640]))
        self.transforms = T.Compose([
            T.Resize(self.size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) if 'DINO' in str(cfg.yaml_cfg) else T.Lambda(lambda x: x)
        ])

    def predict(self, source, conf=0.25, **kwargs):
        im_pil = Image.open(str(source)).convert('RGB')
        w, h = im_pil.size
        orig_size = torch.tensor([[w, h]]).to(self.device)
        im_data = self.transforms(im_pil).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            labels, boxes, scores = self.model(im_data, orig_size)
        
        # 过滤低置信度
        mask = scores[0] > conf
        # 这里的 boxes[0][mask] 是 [N, 4] 的 Tensor
        return [DEIMResults(DEIMBoxes(labels[0][mask], scores[0][mask], boxes[0][mask]))]